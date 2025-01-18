import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import pandas as pd
from measurement_layout_AAIO import *

includeIrrelevantFeatures = True
includeNoise=False
test_synthetic = False
environmentData = dict()
abilityMax = {
    "navigationAbility": 20,
    "visualAbility": 10,
}
abilityMin = {
    "navigationAbility": 0,
    "visualAbility": -999,
}

# FIGURES TO CREATE ON WHICH WE PUT CAPABILITIES

performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
environmentData["abilityMax"] = abilityMax
environmentData["abilityMin"] = abilityMin
all_capabilities = ["ability_navigation", "ability_visual", "ability_bias_rl"]

N = 200  # number of samples
timepoint = 10
excluded_capabilities = []
excluded_capabilities_string = "_".join(excluded_capabilities)
included_capabilities = [c for c in all_capabilities if c not in excluded_capabilities]
df_final = pd.read_csv("eval_results_harder.csv").loc[timepoint*N:(timepoint+1)*N]
df_final = df_final.loc[np.random.choice(df_final.index, N, replace=False)]
successes = df_final["reward"]

successes = successes > -0.9 # SHould be reduced to 1s and 0s

environmentData["reward_distance"] = df_final["reward_distance"].values[0:N]
environmentData["reward_behind"] = df_final["reward_behind"].values[0:N]
environmentData["reward_size"] = df_final["reward_size"].values[0:N]
environmentData["Xpos"] = df_final["Xpos"].values[0:N]
# environmentData["reward_distance"] = df_final["Distance"].values
# environmentData["reward_behind"] = 0
# environmentData["reward_size"] = df_final["Size"].values
# environmentData["Xpos"] = 0
    
    
relevant_figs = [([cap], plt.subplots()) for cap in included_capabilities]



# %%
if __name__ == "__main__":
    m = setupModelSingle(successes, environmentData=environmentData, includeIrrelevantFeatures=includeIrrelevantFeatures, includeNoise=includeNoise, N = N)


    with m:
        inference_data = pm.sample(500, target_accept=0.95, cores=2)
        
    az.plot_posterior(inference_data["posterior"][["ability_navigation", "ability_visual", "ability_bias_rl"]])
    plt.show()
