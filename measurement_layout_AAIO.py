# %%
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import pandas as pd



def rbf_kernel(X, length_scale=1.0):
    """
    Generates a covariance matrix using the RBF (squared exponential) kernel.

    Parameters:
    X (numpy.ndarray): The input data, a 1D array of points.
    length_scale (float): The length scale parameter for the RBF kernel.

    Returns:
    numpy.ndarray: The covariance matrix.
    """
    # Calculate the pairwise squared Euclidean distances
    sq_dists = np.subtract.outer(X, X) ** 2

    # Compute the covariance matrix using the RBF kernel
    K = np.exp(-sq_dists / (2 * length_scale ** 2))

    return K

def margin(a, b):
  return a - b

def scaledBeta(name, a, b, min, max, shape=None):

  beta = pm.Beta(f"{name}_raw", a, b, shape = shape)
  print(beta.shape)
  return pm.Deterministic(name, beta * (max - min) + min)

def setupModel(taskResults, environmentData, cholesky=None, includeIrrelevantFeatures=True, includeNoise=True, N = 200, exclude = []):
  m = pm.Model() # we need to exlcude the navigation performacne itslef and the navigation ability/visual performance nad ability when we apply something to the exclude argument.
  
  # Possible.
  assert taskResults.shape[1] == N
  with m:
    ### Environment Variables as Deterministic
    abilityMax = environmentData["abilityMax"]
    abilityMin = environmentData["abilityMin"]
    demands_distance = pm.MutableData("rewardDistance", environmentData["reward_distance"])
    demands_size = pm.MutableData("rewardSize", environmentData["reward_size"])
    demands_behind = pm.MutableData("rewardBehind", environmentData["reward_behind"])
    performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
    product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
    T = taskResults.shape[0]
    # Some variables need to be initialised
    '''
    
    sigma_nav = 0
    sigma_vis = 0
    sigma_bias = 0
    ability_nav = 0
    ability_visual = 0
    ability_bias_rl = 0
    '''
    # Priors
    sigma_performance = 0
    if includeNoise:
      sigma_performance = pm.Uniform("sigma_noise", lower=0, upper=1)
    if "ability_navigaton" not in exclude:
      sigma_nav = pm.HalfNormal("sigma_nav", sigma=1.0)
      ability_nav = pm.GaussianRandomWalk("ability_navigation", mu = 0, sigma = sigma_nav, shape = T)
    if "ability_bias_rl" not in exclude:
      sigma_vis = pm.HalfNormal("sigma_vis", sigma=1.0)
      ability_visual = pm.GaussianRandomWalk("ability_visual", mu = 0, sigma = sigma_vis, shape = T)
    if (includeIrrelevantFeatures) :
      sigma_bias = pm.HalfNormal("sigma_bias", sigma=1.0)
      ability_bias_rl = pm.GaussianRandomWalk("ability_bias_rl", mu = 0, sigma = sigma_bias, shape = T)                  # [-inf,inf] A parameter to determine whether left or right have an influence. It's expected to be zero, but negative would be left influence and positive a right influence (or vice versa :-)
      abilityMin["rightLeftBias"] = -np.inf
      abilityMax["rightLeftBias"] = np.inf

      demands_xpos = pm.MutableData("Xpos", environmentData["Xpos"])  #

    if (includeIrrelevantFeatures) :
      rightLeftEffect = pm.Deterministic("rightLeftEffect", product_on_time_varying(ability_bias_rl, demands_xpos))
    else :
      rightLeftEffect = 0

    # Performance
    navigation_performance = 1
    visual_performance = 1
    # Now we overwrite if not excluding
    if "ability_navigation" not in exclude:
      navigation_performance = pm.Deterministic("navigation_performance", logistic(performance_from_capability_and_demand_batch(ability_nav, demands_distance*(1/2 * demands_behind + 1)) + rightLeftEffect))
    if "ability_visual" not in exclude:
      visual_performance = pm.Deterministic("visual_performance", logistic(performance_from_capability_and_demand_batch(ability_visual, np.log(demands_distance) - np.log(demands_size))))
    task_performance = pm.Bernoulli("task_performance", p=(1 - sigma_performance)*(navigation_performance*visual_performance) + sigma_performance*0.5, observed=taskResults)
  return m

def setupModelSingle(taskResults, environmentData, includeIrrelevantFeatures=True, includeNoise=True, N = 200, exclude = []):
  m = pm.Model()
  with m:
    ### Environment Variables as Deterministic
    abilityMax = environmentData["abilityMax"]
    abilityMin = environmentData["abilityMin"]
    rewardDistance = pm.MutableData("rewardDistance", environmentData["reward_distance"])
    rewardSize = pm.MutableData("rewardSize", environmentData["reward_size"])
    rewardBehind = pm.MutableData("rewardBehind", environmentData["reward_behind"])

    ### priors
    navigationAbility = scaledBeta("ability_navigation", 1,1, abilityMin["navigationAbility"], abilityMax["navigationAbility"])
    visualAbility = scaledBeta("ability_visual", 1, 1, abilityMin["visualAbility"], abilityMax["visualAbility"])


    if (includeIrrelevantFeatures) :
      rightLeftBias = pm.Normal("ability_bias_rl", 0,1)                  # [-inf,inf] A parameter to determine whether left or right have an influence. It's expected to be zero, but negative would be left influence and positive a right influence (or vice versa :-)
      abilityMin["rightLeftBias"] = -np.inf
      abilityMax["rightLeftBias"] = np.inf

      XPos = pm.MutableData("Xpos", environmentData["Xpos"])  #


      if (includeIrrelevantFeatures) :
        rightLeftEffect = pm.Deterministic("rightLeftEffect", rightLeftBias * XPos)
      else :
        rightLeftEffect = pm.Deterministic("rightLeftEffect", 0)

    ##Performance
    #navigationP = pm.Deterministic("navigationP", logistic(margin(navigationAbility, rewardDistance)))
    navigationP = pm.Deterministic("navigationP", logistic(margin(navigationAbility,  (rewardDistance*(rewardBehind*0.5+1.0)) + rightLeftEffect)))  # Including rewardBehind (multiplies by 1.5 if behind) and rightLeftEffect
    visualP = pm.Deterministic("visual_performance", logistic(margin(visualAbility, np.log(rewardDistance/rewardSize))))


    if includeNoise :
      # noise = 0.5 # Works almost perfectly, except for the perfect model
      noise = 1 - np.mean(taskResults)  # With this noise is complementary to result prior.
      # pm.Uniform("noise", 0, 1) # Doesn't work because it can just adjust this to be 1 or 0 for perfect or worst agents even if there's no noise
      # noise= random.uniform(0.0, 1.0) # Not sure it will generate a value each time.
      #noise= pm.Bernoulli("noise", np.mean(taskResults))  # This would generate values either 0 and 1 with the prior of taskResults
      noiseLevel = pm.Uniform("noiseLevel", 0, 1)
      abilityMax["noiseLevel"] = 1
      abilityMin["noiseLevel"] = 0
      finalP = pm.Deterministic("finalP", (1-noiseLevel)*navigationP*visualP + (noiseLevel*noise))
      taskPerformance=pm.Bernoulli("taskPerformance", finalP, observed=taskResults)
    else:
      taskPerforamnce = pm.Bernoulli("taskPerformance",  navigationP * visualP, observed = taskResults)
  return m

def logistic(x):
  return 1 / (1 + np.exp(-x))
