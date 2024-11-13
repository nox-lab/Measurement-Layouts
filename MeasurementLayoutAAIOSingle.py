import pymc as pm
import numpy as np
import arviz as az
import random as rm
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from typing import Callable
import numpy.typing as npt
import graphviz


includeIrrelevantFeatures = True
includeNoise=False

environmentData = dict()
abilityMax = {
    "navigationAbility": 5.3,
    "visualAbility": 1.9,
}
abilityMin = {
    "navigationAbility": 0,
    "visualAbility": 0,
}

def margin(a, b):
  return a - b

def scaledBeta(name, a, b, min, max):

  beta = pm.Beta(f"{name}_raw", a, b)
  return pm.Deterministic(name, beta * (max - min) + min)

def logistic(x):
  return 1 / (1 + np.exp(-x))

def setupModel(taskResults):
  m = pm.Model()
  with m:
      ### Environment Variables as Deterministic
      rewardDistance = pm.MutableData("rewardDistance", environmentData["reward_distance"])
      rewardSize = pm.MutableData("rewardSize", environmentData["reward_size"])
      rewardBehind = pm.MutableData("rewardBehind", environmentData["reward_behind"])

      ### priors
      navigationAbility = scaledBeta("navigationAbility", 1,1, abilityMin["navigationAbility"], abilityMax["navigationAbility"])
      visualAbility = scaledBeta("visualAbility", 1, 1, abilityMin["visualAbility"], abilityMax["visualAbility"])


      if (includeIrrelevantFeatures) :
        rightLeftBias = pm.Normal("rightLeftBias", 0,1)                  # [-inf,inf] A parameter to determine whether left or right have an influence. It's expected to be zero, but negative would be left influence and positive a right influence (or vice versa :-)
        abilityMin["rightLeftBias"] = -np.inf
        abilityMax["rightLeftBias"] = np.inf

        XPos = pm.MutableData("Xpos", environmentData["Xpos"])  #


        if (includeIrrelevantFeatures) :
          rightLeftEffect = pm.Deterministic("rightLeftEffect", rightLeftBias * XPos)
        else :
          rightLeftEffect = 0

      ##Performance
      #navigationP = pm.Deterministic("navigationP", logistic(margin(navigationAbility, rewardDistance)))
      navigationP = pm.Deterministic("navigationP", logistic(margin(navigationAbility, rewardDistance*(rewardBehind*0.5+1.0)) + rightLeftEffect))  # Including rewardBehind (multiplies by 1.5 if behind) and rightLeftEffect
      visualP = pm.Deterministic("visualP", logistic(margin(visualAbility, rewardSize)))


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


if __name__ == "__main__":
  N = 5000  # number of samples/number of arenas.
  performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
  np.random.seed(0)
  capability_nav = 3.8 # end of training estimates
  capability_vis = 1.4 #
  capability_bias = 1
  # Task capability creation, representing a range of arenas
  behind = np.random.choice([0, 0.5, 1], N) # 0 if in front of agent's front facing direction, 0.5 if l/r of agent's front facing direction, 1 if behind agent's front facing direction
  distance = np.random.uniform(0.1, 5.3, N)
  environmentData["reward_distance"] = distance
  environmentData["reward_behind"] = behind
  # distance = np.random.uniform(0, 5.3, N)
  xpos = np.random.choice([-1, 0, 1], N) # -1 if l of agent's actual position, 0 if in line with agent's actual position, 1 if r of agent's actual position
  reward_size = np.random.uniform(0, 1.9, N)
  environmentData["reward_size"] = reward_size
  environmentData["Xpos"] = xpos
  rightlefteffect_ = capability_bias*xpos
  perf_nav = logistic(capability_nav - (distance*(1/2 * behind + 1) - rightlefteffect_))
  perf_vis = logistic(capability_vis - reward_size)
  successes = np.random.binomial(1, perf_nav*perf_vis, N) == 1

  m = setupModel(successes)

  with m:
    inferenceData = pm.sample(500, target_accept=0.95, cores=2)
  az.plot_posterior(inferenceData["posterior"][["navigationAbility", "visualAbility", "rightLeftBias"]])
  plt.show()