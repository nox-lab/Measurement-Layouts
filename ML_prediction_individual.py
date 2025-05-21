from measurement_layout_incremental import incremental_measurement_layout
import pandas as pd
import os
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from predictive_accuracy import prediction_accuracy
import re
import json
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION, Measurement_Layout_AAIO_precise, Measurement_Layout_AAIO_SMOOTH

capabilities_list = ["Navigation", "Visual", "Bias", "Noise"]
measurement_layout_used = Measurement_Layout_AAIO
added_folder = ""

if "navigation" in capabilities_list and measurement_layout_used == Measurement_Layout_AAIO_NO_NAVIGATION:
    print("Navigation is in capabilities list, but the measurement layout does not include navigation.")
    raise ValueError
if added_folder != "" and measurement_layout_used == Measurement_Layout_AAIO:
    print("Measurement layout is default, but an added folder is specified.")
    raise ValueError
    
noise_level = np.array([0, 0.4])
capabilities_path = rf"estimated_capabilities/"
N_eval = 200
filename = r"progression_model_results_400k_camera"
incremental_estimator = incremental_measurement_layout(N_eval, capabilities_path, filename=filename, noise_level=noise_level)
# incremental_estimator.conditional_min_array_tests()
# incremental_estimator.real_capabilities(measurement_layout_used, capabilities_list)
#raycasts + framestacking might be worth doing.
N_predict = 400
folder_name = filename
precise = True
full_ML = False

layout = measurement_layout_used

i = 0

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

log_dir = fr"./logs/{filename}/" # should end in a /
files = os.listdir(log_dir)
files = [f for f in files]
files = sorted_alphanumeric(files)
print("Files in logs directory:", files)


# renaming files
# num_training_steps = 25000
# i = 2
# for file in files:
#     model_trainingsteps = num_training_steps*i
#     file_path_name = f"{log_dir}/{file}"
#     os.rename(file_path_name, f"./logs/raycasts_with_frame_stacking.zip")
#     i += 1

all_brier_scores = {
    "baseline": [],
    "model": [],
    "XGBOOST": [],
}
added_config_type = "very_precise"
print(files)
for model in files:
    if i == 40:
        break
    model_name = f"{log_dir}{model}"[6:-4] # remove the .zip
    print(model_name)
    print(f"Model {model_name} testing for predictive accuracy.")
    brier_score, brier_score_XGBOOST, brier_score_baseline = prediction_accuracy(layout, folder_name, added_folder, model_name, N_predict, seed = 2, load_eval = False, config_modifier = added_config_type,
                        noise_level = noise_level, cap_time = i, N_eval = N_eval, full=full_ML)
    all_brier_scores["baseline"].append(brier_score_baseline)
    all_brier_scores["model"].append(brier_score)
    all_brier_scores["XGBOOST"].append(brier_score_XGBOOST)
    print(f"Brier score for model {model_name} is {brier_score}.")
    print(f"Brier score for XGBOOST model {model_name} is {brier_score_XGBOOST}.")
    print(f"Brier score for baseline model {model_name} is {brier_score}")
    i += 1

# fig, ax = plt.subplots()

# for key in all_brier_scores:
#     ax.plot(range(len(all_brier_scores[key])), all_brier_scores[key], marker = "o", linestyle= "-", label = "Brier score for " + key)
# ax.set_title("Brier scores for different models")
# ax.set_xlabel("Time")
# ax.set_ylabel("Brier score")
# ax.legend()
# fig.savefig(f"brier_scores_{filename}.png")
# print("Brier scores saved.")
# if os.path.exists(f"all_brier_scores_{filename}.json"):
#     current_brier_scores = json.load(open(f"all_brier_scores_{filename}.json", "r"))
#     for key in all_brier_scores:
#         if key == "model" and full_ML:
#             current_brier_scores["model_full"] = all_brier_scores[key]
#             continue
#         current_brier_scores[key] = all_brier_scores[key]
    
# json.dump(current_brier_scores, open(f"all_brier_scores_{filename}.json", "w"))