import pymc as pm
import numpy as np
import arviz as az
import random as rm
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import graphviz

print(f"Running on PyMC v{pm.__version__}")
     

def logistic(x):
  return 1 / (1 + np.exp(-x))

def scaledBeta(name, a, b, min, max):

  beta = pm.Beta(f"{name}_raw", a, b)
  return pm.Deterministic(name, beta * (max - min) + min)


def logistic(x):
  return 1 / (1 + np.exp(-x))

def scaledBeta(name, a, b, min, max):

  beta = pm.Beta(f"{name}_raw", a, b)
  return pm.Deterministic(name, beta * (max - min) + min)
     

df = pd.read_csv("https://raw.githubusercontent.com/Kinds-of-Intelligence-CFI/measurement-layout-tutorial/main/data/visionNavigationMeasurementLayoutData.csv")
     

df.describe()

def setupModel(relevantData):


  m = pm.Model()



  with  m:
    instanceMetafeatureDistance = pm.MutableData("InstanceDistances", relevantData["Distance"])
    instanceMetafeatureSize = pm.MutableData("InstanceSizes", relevantData["Size"])
    navigationAbility = scaledBeta("NavigationAbility", 1,1, 0, 200)
    visualAbility = scaledBeta("VisualAbility", 1,1, 0, 1000)

    navigationP = pm.Deterministic("navigationP", logistic(navigationAbility - instanceMetafeatureDistance))
    visualP = pm.Deterministic("visualP", logistic(np.log(visualAbility) - np.log(instanceMetafeatureDistance/instanceMetafeatureSize)))


    finalP = pm.Deterministic("FinalP", navigationP*visualP)
    observed = pm.Bernoulli("ObservedPerformance", finalP, observed =relevantData["Success"])
  return m
     
if __name__ == "__main__":
    m = setupModel(df)
    with m:
        inferenceData = pm.sample(1000, target_accept=0.95, cores=2)
    
    az.plot_posterior(inferenceData["posterior"][["NavigationAbility", "VisualAbility"]])
    plt.show()
