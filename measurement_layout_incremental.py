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
import os

# X is navigation, visual, bias, [distance, szie, behind, x_pos]
class incremental_measurement_layout():
  def __init__(self, N, folder, filename):
    # Folder will specify where to save teh 
    self.N = N
    self.folder = folder
    self.noise_level = 0.0
    self.filename = filename
    self.environmentData = pd.read_csv(rf"csv_recordings/{filename}.csv")
    rewards_from_evaluation = self.environmentData["reward"].to_numpy()
    self.T = int(len(rewards_from_evaluation)/N)  # number of time steps
    self.noisy_model_performance = np.sum(rewards_from_evaluation[-N:] > -0.9)/N
    self.overall_performance = np.reshape(rewards_from_evaluation, (self.T, N))
    self.overall_successes = np.average(self.overall_performance > -0.9, axis = 1) # Sum over N
    
  def logistic(self, x):
      return 1 / (1 + np.exp(-x))
    
  def base_test(self, layout):
    performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
    product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
    np.random.seed(0)
    T = self.T 
    learn_time_nav = 0.5*T
    learn_time_vis = 0.2*T
    learn_time_bias = 0.3*T
    time_steps = np.linspace(1, T, T)
    capability_nav = self.logistic((time_steps - learn_time_nav)/(T/5))*5.6 #particular point where significant learning occurs, and rate at which this is is determined by the denominator
    capability_vis = self.logistic((time_steps - learn_time_vis)/(T/5))*1.9
    capability_bias = self.logistic((time_steps- learn_time_bias)/(T/5))*1
    
    xpos = np.random.choice([-1, 0, 1], self.N)
    distance = np.random.uniform(0, 5.6, self.N)
    reward_size = np.random.uniform(0, 1.9, self.N)
    behind = np.random.choice([0, 0.5, 1], self.N)
    
    # Task capability creation, representing a range of arenas
    rightlefteffect_ = product_on_time_varying(capability_bias, xpos)
    perf_nav = self.logistic(performance_from_capability_and_demand_batch(capability_nav, distance*(behind*0.5+1.0)) + rightlefteffect_)
    perf_vis = self.logistic(performance_from_capability_and_demand_batch(capability_vis, np.log(distance/reward_size)))
    self.successes = np.random.binomial(1, perf_nav*perf_vis, (T, self.N)) == 1
    self.estimate_capabilities(layout, cap_labels = ["nav", "visual", "bias"])
    print("Base test done")

  def real_capabilities(self, layout, cap_labels = ["nav", "visual", "bias"]):
    T = self.T
    successes = self.environmentData["reward"].to_numpy() > -0.9
    
    print(successes.shape)
    successes = successes.astype(int)
    self.successes = successes.reshape((T, self.N))
    self.estimate_capabilities(layout, cap_labels)

  def estimate_capabilities(self, layout, cap_labels = ["nav", "visual", "bias"]):
      
    filename = self.filename
    folder = self.folder
    layout = layout(self.N, self.environmentData, noiselevel = self.noise_level, noisy_model_performance = self.noisy_model_performance)
    fk_model = ssm.Bootstrap(ssm=layout, data=self.successes)
    my_pmmh = SMC(N=500, fk=fk_model, collect = [Moments()])
    my_pmmh.run()
    processed_chain = my_pmmh.summaries.moments
    capability_profiles = dict()
    if not os.path.exists(rf"{folder}/{filename}"):
      os.makedirs(rf"{folder}/{filename}")
    for i in range(len(cap_labels)):
      capability_profiles[cap_labels[i]] = dict()
      capability_profiles[cap_labels[i]]["mean"] = [mom["mean"][i] for mom in processed_chain]
      capability_profiles[cap_labels[i]]["var"] = [mom["var"][i] for mom in processed_chain]
      np.save(rf"{folder}/{filename}/{cap_labels[i]}_est", capability_profiles[cap_labels[i]]["mean"])
      np.save(rf"{folder}/{filename}/{cap_labels[i]}_var", capability_profiles[cap_labels[i]]["var"])
      
    
    time_steps = np.linspace(1, self.T, self.T)
    T = self.T
    num_caps = len(processed_chain[0]["mean"])/2 # Number of mean vectors we have
    num_rows = int(num_caps//2 + 1)
    fig3, ax3 = plt.subplots(num_rows, 2, figsize=(10, 6))
    counter = 0
    for i in range(num_rows):
      for j in range(2):
        if i == num_rows - 1 and j == 1:
          continue
        if counter >= num_caps:
          break
        ax3[i, j].plot(time_steps, [mom["mean"][counter] for mom in processed_chain], label=cap_labels[counter])
        ax3[i, j].legend()
        counter += 1
    ax3[num_rows - 1,1].bar(time_steps, self.overall_successes, label="Success rate")
    ax3[num_rows - 1 ,1].set_xlabel("Time")
    ax3[num_rows - 1,1].set_ylabel("Success rate")
    fig3.savefig(rf"{folder}/{filename}/estimated_capabilities.png")
    #print(np.mean(processed_chain, 0))
