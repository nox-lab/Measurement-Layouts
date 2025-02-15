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
from pandas import DataFrame
import os

# X is navigation, visual, bias, [distance, szie, behind, x_pos]


# Now with variable variance !
class Measurement_Layout_AAIO(ssm.StateSpaceModel):
    def __init__(self, N : int, environmentData : DataFrame, sigmanav = 10, sigmavis = 10, sigmabias = 10, noiselevel = 0, noisy_model_performance = 0):
        self.sigmanav = sigmanav
        self.sigmavis = sigmavis
        self.sigmabias = sigmabias
        self.distance = environmentData["reward_distance"][:N].to_numpy()
        self.behind = environmentData["reward_behind"][:N].to_numpy()
        self.reward_size = environmentData["reward_size"][:N].to_numpy()
        self.xpos = environmentData["Xpos"][:N].to_numpy()
        self.noiselevel = noiselevel
        self.noisy_model_performance = noisy_model_performance
        self.N = N
        
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))
        

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
        product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
        performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
        rightlefteffect = product_on_time_varying(x[:, 2], self.xpos)
        perf_nav = self.logistic(performance_from_capability_and_demand_batch(x[:, 0], self.distance*(self.behind*0.5+1.0)) + rightlefteffect)
        perf_vis = self.logistic(performance_from_capability_and_demand_batch(x[:, 1], np.log(self.distance/self.reward_size)))
        final_prob = self.noiselevel*self.noisy_model_performance + (1-self.noiselevel) * perf_nav * perf_vis
        final_prob = np.swapaxes(final_prob, 0, 1)
        observations_kwargs = tuple([dists.Binomial(1, p = final_prob[_]) for _ in range(self.N)])
        return dists.IndepProd(*observations_kwargs)
    

# Now with variable variance !
class Measurement_Layout_AAIO_NO_NAVIGATION(ssm.StateSpaceModel):
    def __init__(self, N : int, environmentData : DataFrame, sigmavis = 10, sigmabias = 10, noiselevel = 0, noisy_model_performance = 0):
        self.sigmavis = sigmavis
        self.sigmabias = sigmabias
        self.distance = environmentData["reward_distance"][:N].to_numpy()
        self.behind = environmentData["reward_behind"][:N].to_numpy()
        self.reward_size = environmentData["reward_size"][:N].to_numpy()
        self.xpos = environmentData["Xpos"][:N].to_numpy()
        self.noiselevel = noiselevel
        self.noisy_model_performance = noisy_model_performance
        self.N = N
        
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))
        

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=0, scale = self.sigmavis),
                               dists.Normal(loc=0, scale = self.sigmabias),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale = xp[:, 2]),
                               dists.Normal(loc=xp[:, 1], scale = xp[:, 3]),
                               dists.Dirac(loc=xp[:, 2]),
                               dists.Dirac(loc=xp[:, 3]),
                               )

    def PY(self, t, xp, x):
        product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
        performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
        rightlefteffect = product_on_time_varying(x[:, 1], self.xpos)
        perf_vis = self.logistic(performance_from_capability_and_demand_batch(x[:, 0], np.log(self.distance/self.reward_size) + rightlefteffect))
        final_prob = self.noiselevel*self.noisy_model_performance + (1-self.noiselevel) * perf_vis
        final_prob = np.swapaxes(final_prob, 0, 1)
        observations_kwargs = tuple([dists.Binomial(1, p = final_prob[_]) for _ in range(self.N)])
        return dists.IndepProd(*observations_kwargs)