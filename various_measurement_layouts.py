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
    # Changing the priors leads to very different capability estimates. 
    def __init__(self, N : int, environmentData : DataFrame, sigmanav = 1, sigmavis = 1, sigmabias = 1, noiselevel: np.ndarray = np.array([0, 1]), noisy_model_performance = 0):
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
        self.T = environmentData["reward_distance"].shape[0]/N
        self.arena_outcomes = []
        self.noise_max = noiselevel[1]
        self.noise_min = noiselevel[0]
        
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))
        

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=0, scale = self.sigmanav),
                               dists.Normal(loc=0, scale = self.sigmavis),
                               dists.Normal(loc=0, scale = self.sigmabias),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.Uniform(a=self.noise_min, b=self.noise_max),
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale = xp[:, 3]),
                               dists.Normal(loc=xp[:, 1], scale = xp[:, 4]),
                               dists.Normal(loc=xp[:, 2], scale = xp[:, 5]),
                               dists.TruncNormal(mu=xp[:, 3], sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=xp[:, 4], sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=xp[:, 5], sigma = 1, a = 0, b = 99),
                               dists.Uniform(a=self.noise_min, b=self.noise_max),
                               )

    def PY(self, t, xp, x):
        product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
        performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
        rightlefteffect = product_on_time_varying(x[:, 2], self.xpos)
        perf_nav = self.logistic(performance_from_capability_and_demand_batch(x[:, 0], self.distance*(self.behind*0.5+1.0)) + rightlefteffect)
        perf_vis = self.logistic(performance_from_capability_and_demand_batch(x[:, 1], np.log(self.distance/self.reward_size)))
        
        final_prob = x[:, 6][:, None]*self.noisy_model_performance + (1-x[:, 6][:, None]) * perf_nav * perf_vis
        final_prob = np.swapaxes(final_prob, 0, 1)
        observations_kwargs = tuple([dists.Binomial(1, p = final_prob[_]) for _ in range(self.N)])
        self.arena_outcomes = final_prob
        #print(np.c_[np.arange(len(final_prob)), final_prob, np.ravel(perf_nav), np.ravel(perf_vis)])
        return dists.IndepProd(*observations_kwargs)
    

class Measurement_Layout_AAIO_precise(ssm.StateSpaceModel):
    def __init__(self, N : int, environmentData : DataFrame, sigmanav = 10, sigmavis = 10, sigmabias = 10, noiselevel = 0, noisy_model_performance = 0):
        self.sigmanav = sigmanav
        self.sigmavis = sigmavis
        self.sigmabias = sigmabias
        self.distance = environmentData["reward_distance"][:N].to_numpy()
        self.behind = environmentData["reward_behind"][:N].to_numpy()
        self.reward_size = environmentData["reward_size"][:N].to_numpy()
        self.xpos = environmentData["Xpos"][:N].to_numpy()
        self.angles = np.zeros(N)
        for i in range(N):
            if self.xpos[i] == 0 and self.behind[i] == 0:
                self.angles[i] = 0
            elif self.xpos[i] == 1 and self.behind[i] == 0:
                self.angles[i] = 45
            elif self.xpos[i] == -1 and self.behind[i] == 0:
                self.angles[i] = -45
            elif self.xpos[i] == 1 and self.behind[i] == 0.5:
                self.angles[i] = 90
            elif self.xpos[i] == -1 and self.behind[i] == 0.5:
                self.angles[i] = -90
            elif self.xpos[i] == 0 and self.behind[i] == 1:
                self.angles[i] = 180
            elif self.xpos[i] == 1 and self.behind[i] == 1:
                self.angles[i] = 135
            elif self.xpos[i] == -1 and self.behind[i] == 1:
                self.angles[i] = -135
        self.noiselevel = noiselevel
        self.noisy_model_performance = noisy_model_performance
        self.N = N
        self.arena_outcomes = []
        
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))
        

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=0, scale = self.sigmanav),
                               dists.Normal(loc=0, scale = self.sigmavis),
                               dists.Normal(loc=0, scale = self.sigmabias),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.Uniform(a=0, b=1)
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale = xp[:, 3]),
                               dists.Normal(loc=xp[:, 1], scale = xp[:, 4]),
                               dists.Normal(loc=xp[:, 2], scale = xp[:, 5]),
                               dists.Dirac(loc=xp[:, 3]),
                               dists.Dirac(loc=xp[:, 4]),
                               dists.Dirac(loc=xp[:, 5]),
                               dists.Uniform(a=0, b=1)
                               )

    def PY(self, t, xp, x):
        product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
        performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
        extent_of_turning = np.abs(self.angles/180) # 1 for full turn and variation depending on extent of turn.
        extent_of_lr = np.sin(self.angles*np.pi/180)
        rightlefteffect = product_on_time_varying(x[:, 2], extent_of_lr)
        perf_nav = self.logistic(performance_from_capability_and_demand_batch(x[:, 0], self.distance*(1+extent_of_turning*1) + rightlefteffect))
        perf_vis = self.logistic(performance_from_capability_and_demand_batch(x[:, 1], np.log(self.distance/self.reward_size)))
        final_prob = x[:, 6]*self.noisy_model_performance + (1-x[:, 6]) * perf_nav * perf_vis
        final_prob = np.swapaxes(final_prob, 0, 1)
        self.arena_outcomes = final_prob
        observations_kwargs = tuple([dists.Binomial(1, p = final_prob[_]) for _ in range(self.N)])
        return dists.IndepProd(*observations_kwargs)

# Now with variable variance !
class Measurement_Layout_AAIO_NO_NAVIGATION(ssm.StateSpaceModel):
    def __init__(self, N : int, environmentData : DataFrame, sigmavis = 10, sigmabias = 10, noiselevel: np.ndarray = np.array([0, 1]), noisy_model_performance = 0):
        self.sigmavis = sigmavis
        self.sigmabias = sigmabias
        self.distance = environmentData["reward_distance"][:N].to_numpy()
        self.behind = environmentData["reward_behind"][:N].to_numpy()
        self.reward_size = environmentData["reward_size"][:N].to_numpy()
        self.xpos = environmentData["Xpos"][:N].to_numpy()
        self.noiselevel = noiselevel
        self.noisy_model_performance = noisy_model_performance
        self.N = N
        self.arena_outcomes = []
        self.noise_max = noiselevel[1]
        self.noise_min = noiselevel[0]
        
    def logistic(self, x):
        boolean_mask = x > -50
        new_x = np.zeros(x.shape)
        new_x[boolean_mask] = 1 / (1 + np.exp(-x[boolean_mask]))
        return new_x
        

    def PX0(self):
        return dists.IndepProd(dists.Normal(loc=0, scale = self.sigmavis),
                               dists.Normal(loc=0, scale = self.sigmabias),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.TruncNormal(mu=0, sigma = 1, a = 0, b = 99),
                               dists.Uniform(a=self.noise_min, b=self.noise_max),
                               )

    def PX(self, t, xp):
        return dists.IndepProd(dists.Normal(loc=xp[:, 0], scale = xp[:, 2]),
                               dists.Normal(loc=xp[:, 1], scale = xp[:, 3]),
                               dists.TruncNormal(mu=xp[:, 2], sigma = 0.5, a = 0, b = 99),
                               dists.TruncNormal(mu=xp[:, 3], sigma = 0.5, a = 0, b = 99),
                               dists.Uniform(a=self.noise_min, b=self.noise_max),
                               )

    def PY(self, t, xp, x):
        product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
        performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
        rightlefteffect = product_on_time_varying(x[:, 1], self.xpos)
        perf_vis = self.logistic(x[:, 0][:, None] - np.log(self.distance/self.reward_size) + rightlefteffect)
        final_prob = x[:, 4][:, None]*self.noisy_model_performance + (1-x[:, 4][:, None]) * perf_vis
        final_prob = np.swapaxes(final_prob, 0, 1)
        observations_kwargs = tuple([dists.Binomial(1, p = final_prob[_]) for _ in range(self.N)])
        self.arena_outcomes = final_prob
        return dists.IndepProd(*observations_kwargs)
    
    
class Measurement_Layout_AAIO_SMOOTH(ssm.StateSpaceModel):
    def __init__(self, N: int, environmentData: DataFrame, sigmanav=10, sigmavis=10, sigmabias=10, 
                 noiselevel: np.ndarray = np.array([0, 1]), noisy_model_performance=0, prev_capability_weight=0.5):
        self.sigmanav = sigmanav
        self.sigmavis = sigmavis
        self.sigmabias = sigmabias
        self.prev_capability_weight = prev_capability_weight
        self.distance = environmentData["reward_distance"][:N].to_numpy()
        self.behind = environmentData["reward_behind"][:N].to_numpy()
        self.reward_size = environmentData["reward_size"][:N].to_numpy()
        self.xpos = environmentData["Xpos"][:N].to_numpy()
        self.noiselevel = noiselevel
        self.noisy_model_performance = noisy_model_performance
        self.N = N
        self.T = environmentData["reward_distance"].shape[0] / N
        self.arena_outcomes = []
        self.noise_max = noiselevel[1]
        self.noise_min = noiselevel[0]
        
    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def PX0(self):
        return dists.IndepProd(
            dists.Normal(loc=0, scale=self.sigmanav),
            dists.Normal(loc=0, scale=self.sigmanav),
            dists.Normal(loc=0, scale=self.sigmavis),
            dists.Normal(loc=0, scale=self.sigmavis),
            dists.Normal(loc=0, scale=self.sigmabias),
            dists.Normal(loc=0, scale=self.sigmabias),
            dists.TruncNormal(mu=0, sigma=1, a=0, b=99),
            dists.TruncNormal(mu=0, sigma=1, a=0, b=99),
            dists.TruncNormal(mu=0, sigma=1, a=0, b=99),
            dists.Uniform(a=self.noise_min, b=self.noise_max),
            dists.Normal(loc=0, scale=self.sigmanav),  # Previous nav capability
            dists.Normal(loc=0, scale=self.sigmavis)   # Previous vis capability
        )
    
    def PX(self, t, xp):
        return dists.IndepProd(
            dists.Normal(loc=xp[:, 0], scale=xp[:, 3]),
            dists.Normal(loc=xp[:, 1], scale=xp[:, 4]),
            dists.Normal(loc=xp[:, 2], scale=xp[:, 5]),
            dists.TruncNormal(mu=xp[:, 3], sigma=0.5, a=0, b=99),
            dists.TruncNormal(mu=xp[:, 4], sigma=0.5, a=0, b=99),
            dists.TruncNormal(mu=xp[:, 5], sigma=0.5, a=0, b=99),
            dists.Uniform(a=self.noise_min, b=self.noise_max),
            dists.Normal(loc=self.prev_capability_weight * xp[:, 0] + (1 - self.prev_capability_weight) * xp[:, 10], scale=self.sigmanav),
            dists.Normal(loc=self.prev_capability_weight * xp[:, 1] + (1 - self.prev_capability_weight) * xp[:, 11], scale=self.sigmavis)
        )
    
    def PY(self, t, xp, x):
        product_on_time_varying = lambda capability, demand: (capability[:, None] * demand)
        performance_from_capability_and_demand_batch = lambda capability, demand: (capability[:, None] - demand)
        rightlefteffect = product_on_time_varying(x[:, 2], self.xpos)
        perf_nav = self.logistic(performance_from_capability_and_demand_batch(x[:, 0], self.distance * (self.behind * 0.5 + 1.0)) + rightlefteffect)
        perf_vis = self.logistic(performance_from_capability_and_demand_batch(x[:, 1], np.log(self.distance / self.reward_size)))
        
        final_prob = x[:, 6][:, None] * self.noisy_model_performance + (1 - x[:, 6][:, None]) * perf_nav * perf_vis
        final_prob = np.swapaxes(final_prob, 0, 1)
        observations_kwargs = tuple([dists.Binomial(1, p=final_prob[_]) for _ in range(self.N)])
        self.arena_outcomes = final_prob
        
        return dists.IndepProd(*observations_kwargs)
