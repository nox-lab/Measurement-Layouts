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

# X is navigation, visual, bias, [distance, szie, behind, x_pos]
N = 200

product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
def logistic(x):
    return 1 / (1 + np.exp(-x))
  
# Task capability creation, representing a range of arenas
behind = np.random.choice([0, 0.5, 1], N) # 0 if in front of agent's front facing direction, 0.5 if l/r of agent's front facing direction, 1 if behind agent's front facing direction
distance = np.random.uniform(0.1, 5.3, N)
xpos = np.random.choice([-1, 0, 1], N) # -1 if l of agent's actual position, 0 if in line with agent's actual position, 1 if r of agent's actual position
reward_size = np.random.uniform(0.0, 1.9, N)
reg_prior_dict = OrderedDict()
reg_prior_dict['cap'] = dists.MvNormal(cov=np.eye(3))
reg_prior = dists.StructDist(reg_prior_dict)
class Measurement_Layout_AAIO(ssm.StateSpaceModel):
    default_params = {'sigmanav': 2.e-4,
                      'sigmavis': 1.e-4,
                      'sigmabias': 1.e-3,
                      'x0': np.array([1, 1, 1])
                     }

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=self.x0[0], scale = self.sigmanav),
                               dists.Normal(loc=self.x0[1], scale = self.sigmavis),
                               dists.Normal(loc=self.x0[2], scale = self.sigmabias),
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale = self.sigmanav),
                               dists.Normal(loc=xp[:, 1], scale = self.sigmavis),
                               dists.Normal(loc=xp[:, 2], scale = self.sigmabias),
                               )

    def PY(self, t, xp, x):
        rightlefteffect = product_on_time_varying(x[:, 2], xpos)
        perf_nav = logistic(performance_from_capability_and_demand_batch(x[:, 0], distance*(behind*0.5+1.0)) + rightlefteffect)
        perf_vis = logistic(performance_from_capability_and_demand_batch(x[:, 1], np.log(distance/reward_size)))
        final_prob = perf_nav * perf_vis
        final_prob = final_prob.squeeze()
        observations_kwargs = tuple([dists.Binomial(1, p = final_prob[_]) for _ in range(N)])
        return dists.IndepProd(*observations_kwargs)
      

if __name__ == "__main__":
  
  T = 5  # number of time steps
  performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
  product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
  np.random.seed(0)
  learn_time_nav = 0.5*T
  learn_time_vis = 0.2*T
  learn_time_bias = 0.3*T
  time_steps = np.linspace(1, T, T)
  capability_nav = logistic((time_steps - learn_time_nav)/(T/5))*5.3 #particular point where significant learning occurs, and rate at which this is is determined by the denominator
  capability_vis = logistic((time_steps - learn_time_vis)/(T/5))*1.9
  capability_bias = logistic((time_steps- learn_time_bias)/(T/5))
  # Task capability creation, representing a range of arenas
  np.random.seed(0)
  # Task capability creation, representing a range of arenas
  behind = np.random.choice([0, 0.5, 1], N) # 0 if in front of agent's front facing direction, 0.5 if l/r of agent's front facing direction, 1 if behind agent's front facing direction
  distance = np.random.uniform(0.1, 5.3, N)
  # distance = np.random.uniform(0, 5.3, N)
  xpos = np.random.choice([-1, 0, 1], N) # -1 if l of agent's actual position, 0 if in line with agent's actual position, 1 if r of agent's actual position
  reward_size = np.random.uniform(0.0, 1.9, N)
  rightlefteffect_ = product_on_time_varying(capability_bias, xpos)
  perf_nav = logistic(performance_from_capability_and_demand_batch(capability_nav, distance*(behind*0.5+1.0)) + rightlefteffect_)
  perf_vis = logistic(performance_from_capability_and_demand_batch(capability_vis, np.log(distance/reward_size)))
  successes = np.random.binomial(1, perf_nav*perf_vis, (T, N)) == 1
  my_pmmh = mcmc.PMMH(ssm_cls=Measurement_Layout_AAIO, prior = reg_prior, data=successes, Nx=N,
                    niter=100)
  my_pmmh.run()
  processed_chain = np.array(my_pmmh.chain.theta.tolist())
  print(type(my_pmmh.chain.theta))
  print(type(my_pmmh.chain.theta[0]))
  print(my_pmmh.chain.theta)
  print(processed_chain.shape)
  print(np.mean(processed_chain, 0))
