# %%
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
from pytensor.printing import Print
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import mcmc
from collections import OrderedDict
from particles import SMC
from particles.collectors import Moments
import pandas as pd

# X is navigation, visual, bias, [distance, szie, behind, x_pos]
N = 200

product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
def logistic(x):
    return 1 / (1 + np.exp(-x))
  
df_caps = pd.read_csv(r"csv_recordings/working_caps_predictive_4_harder_train10eval15.csv")
distance = df_caps["reward_distance"][:N].to_numpy()
behind = df_caps["reward_behind"][:N].to_numpy()
reward_size = df_caps["reward_size"][:N].to_numpy()
xpos = df_caps["Xpos"][:N].to_numpy()
rewards_from_evaluation = df_caps["reward"].to_numpy()
T = int(len(rewards_from_evaluation)/N)  # number of time steps
reg_prior_dict = OrderedDict()
reg_prior_dict['sigmanav'] = dists.TruncNormal(mu=0, sigma=1, a = 0, b = 99)
reg_prior_dict['sigmavis'] = dists.TruncNormal(mu=0, sigma=1, a = 0, b = 99)
reg_prior_dict['sigmabias'] = dists.TruncNormal(mu=0, sigma=1, a = 0, b = 99)
reg_prior_dict['x0'] = dists.MvNormal(cov = np.eye(3))
reg_prior = dists.StructDist(reg_prior_dict)
noise_level = 0.1
noisy_model_performance = np.sum(rewards_from_evaluation[-N:] > -0.9)/N
print(len(rewards_from_evaluation)/N)
overall_performance = np.reshape(rewards_from_evaluation, (T, N))
overall_successes = np.average(overall_performance > -0.9, axis = 1) # Sum over N
print(overall_successes.shape)
print(T)

# Now with variable variance !
class Measurement_Layout_AAIO(ssm.StateSpaceModel):
    default_params = {'sigmanav': 1,
                      'sigmavis': 1,
                      'sigmabias': 1,
                     }

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=0, scale = self.sigmanav),
                               dists.Normal(loc=0, scale = self.sigmavis),
                               dists.Normal(loc=0, scale = self.sigmabias),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale = xp[:, 3]),
                               dists.Normal(loc=xp[:, 1], scale = xp[:, 4]),
                               dists.Normal(loc=xp[:, 2], scale = xp[:, 5]),
                               dists.Dirac(loc=xp[:, 3]),
                               dists.Dirac(loc=xp[:, 4]),
                               dists.Dirac(loc=xp[:, 5]),
                               )

    def PY(self, t, xp, x):
        rightlefteffect = product_on_time_varying(x[:, 2], xpos)
        perf_nav = logistic(performance_from_capability_and_demand_batch(x[:, 0], distance*(behind+1.0)) + rightlefteffect)
        perf_vis = logistic(performance_from_capability_and_demand_batch(x[:, 1], np.log(distance/reward_size)))
        final_prob = noise_level*noisy_model_performance + (1-noise_level) * perf_nav * perf_vis
        final_prob = np.swapaxes(final_prob, 0, 1)
        observations_kwargs = tuple([dists.Binomial(1, p = final_prob[_]) for _ in range(N)])
        return dists.IndepProd(*observations_kwargs)
      

if __name__ == "__main__":
  print(T)
  performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
  product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
  np.random.seed(0)
  learn_time_nav = 0.5*T
  learn_time_vis = 0.2*T
  learn_time_bias = 0.3*T
  time_steps = np.linspace(1, T, T)
#   capability_nav = logistic((time_steps - learn_time_nav)/(T/5))*5.6 #particular point where significant learning occurs, and rate at which this is is determined by the denominator
#   capability_vis = logistic((time_steps - learn_time_vis)/(T/5))*1.9
#   capability_bias = logistic((time_steps- learn_time_bias)/(T/5))*1
#   # Task capability creation, representing a range of arenas
#   rightlefteffect_ = product_on_time_varying(capability_bias, xpos)
#   perf_nav = logistic(performance_from_capability_and_demand_batch(capability_nav, distance*(behind*0.5+1.0)) + rightlefteffect_)
#   perf_vis = logistic(performance_from_capability_and_demand_batch(capability_vis, np.log(distance/reward_size)))
  #   successes = np.random.binomial(1, perf_nav*perf_vis, (T, N)) == 1

  successes = df_caps["reward"].to_numpy() > -0.9
  print(successes.shape)
  successes = successes.astype(int)
  successes = successes.reshape((T, N))
  fk_model = ssm.Bootstrap(ssm=Measurement_Layout_AAIO(), data=successes)
  my_pmmh = SMC(N=500, fk=fk_model, collect = [Moments()])
  my_pmmh.run()
  processed_chain = my_pmmh.summaries.moments
  print(processed_chain)
  fig, ax = plt.subplots(1, 1, figsize=(10, 6))
  ax.plot(time_steps, [mom["mean"][0:3] for mom in processed_chain], label=["Estimated navigation", "Estimated visual", "Estimated bias"])
  ax.plot(time_steps, [mom["var"][0:3] + mom["mean"][0:3] for mom in processed_chain], alpha = 0.2)
  ax.plot(time_steps, [mom["mean"][0:3] - mom["var"][0:3] for mom in processed_chain], alpha = 0.2)
  ax2 = ax.twinx()
  ax2.set_ylabel("Success rate")
  ax2.bar(range(T), overall_successes, alpha = 0.2, color = "grey")
  #   ax.plot(time_steps, capability_nav, label="true capability")
  #   ax.plot(time_steps, capability_vis, label="true capability")
  #   ax.plot(time_steps, capability_bias, label="true capability")
  ax.set_xlabel("Time")
  ax.set_ylabel("Capability")   
  ax.legend()
  plt.show()
  #print(np.mean(processed_chain, 0))
