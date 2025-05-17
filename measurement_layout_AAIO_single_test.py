import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import pandas as pd
from measurement_layout_AAIO import *
import os
from various_measurement_layouts import Measurement_Layout_AAIO
from predictive_accuracy import prediction_accuracy
import re
import json

includeIrrelevantFeatures = True
includeNoise=False
test_synthetic = False
environmentData = dict()
abilityMax = {
    "navigationAbility": 35,
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
excluded_capabilities = []
excluded_capabilities_string = "_".join(excluded_capabilities)
included_capabilities = [c for c in all_capabilities if c not in excluded_capabilities]
df_final = pd.read_csv("csv_recordings/progression_model_results_400k_camera.csv")
successes = df_final["reward"]
print(successes)
T = int(len(successes.to_numpy().flatten())/N)
print(f"Number of timepoints = {T}")

successes = successes > -0.9 # SHould be reduced to 1s and 0s
print(f"average successes = {np.average(successes)}")

environmentData["reward_distance"] = df_final["reward_distance"].values[0:N]
environmentData["reward_behind"] = df_final["reward_behind"].values[0:N]
environmentData["reward_size"] = df_final["reward_size"].values[0:N]
environmentData["Xpos"] = df_final["Xpos"].values[0:N]
# environmentData["reward_distance"] = df_final["Distance"].values
# environmentData["reward_behind"] = 0
# environmentData["reward_size"] = df_final["Size"].values
# environmentData["Xpos"] = 0

# Could assess model fit by going over all of the possible posterior samples and seeing how well they fit the data.
# Could also assess the predictive accuracy of the model by looking at the posterior predictive distribution.



# %%
if __name__ == "__main__":
    brier_scores_all = {"baseline": [], "model": []}
    capabilities_for_single = {"ability_navigation": [], "ability_visual": [], "ability_bias_rl": []}
    for timepoint in range(T):
        
        df_final = pd.read_csv("csv_recordings/progression_model_results_400k_camera.csv").loc[timepoint*N:(timepoint+1)*N-1]
        successes = df_final["reward"]

        successes = successes > -0.9 # SHould be reduced to 1s and 0s
        print(successes)
        print(f"average successes = {np.average(successes)}")
        
        
        m = setupModelSingle(successes, environmentData=environmentData, includeIrrelevantFeatures=includeIrrelevantFeatures, includeNoise=includeNoise, N = N)


        with m:
            inference_data = pm.sample(1000, target_accept=0.95, cores=2)

        capabilities_out = inference_data["posterior"][["ability_navigation", "ability_visual", "ability_bias_rl"]].mean()
        print(capabilities_out)

        for i, cap in enumerate(included_capabilities):
            capabilities_for_single[cap].append(capabilities_out[cap])
        # az.plot_posterior(inference_data["posterior"][["ability_navigation", "ability_visual", "ability_bias_rl"]])
        # plt.show()
        
        
        def sorted_alphanumeric(data):
            convert = lambda text: int(text) if text.isdigit() else text.lower()
            alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
            return sorted(data, key=alphanum_key)

        log_dir = "./logs/model_progression_camera"
        files = os.listdir(log_dir)
        files = [f for f in files]
        files = sorted_alphanumeric(files)
        print("Files in logs directory:", files)
        model = files[timepoint]
        model_name = f"{log_dir}/{model}"[6:-4] # remove the .zip
        noise_level = np.array([0, 0.2])
        layout = Measurement_Layout_AAIO
        added_folder = ""
        folder_name = "progression_model_results_400k_camera"
        N_eval = N
        N_predict = 100
        precise = True
        full_ML = True
        print(model_name)
        print(f"Model {model_name} testing for predictive accuracy.")
        brier_score, brier_score_XGBOOST, brier_score_baseline = prediction_accuracy(layout, folder_name, added_folder, model_name, N_predict, load_eval = True, config_modifier="very_precise",
                            noise_level = noise_level, cap_time = timepoint, N_eval = N_eval, caps = capabilities_out, seed = 2)
        print(f"Brier score for model {model_name} is {brier_score}.")
        print(f"Brier score for XGBOOST model {model_name} is {brier_score_XGBOOST}.")
        print(f"Brier score for baseline model {model_name} is {brier_score_baseline}")
        brier_scores_all["baseline"].append(brier_score_baseline)
        brier_scores_all["model"].append(brier_score)
    json.dump(brier_scores_all, open("brier_scores_all_camera_single.json", "w"))
    # for single_capability in capabilities_for_single:
    #     capabilities_for_single[single_capability] = np.array(capabilities_for_single[single_capability])
    #     np.savez(f"estimated_capabilities/camera_with_frame_stacking_400k/{single_capability}_single.npz", capabilities_for_single[single_capability])
