from measurement_layout_incremental import incremental_measurement_layout
import pandas as pd
import os
import numpy as np
import numpy.typing as npt
from predictive_accuracy import prediction_accuracy
import re
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION, Measurement_Layout_AAIO_precise, Measurement_Layout_AAIO_SMOOTH

capabilities_list = ["navigation", "visual", "bias", "noise"]
measurement_layout_used = Measurement_Layout_AAIO_SMOOTH
added_folder = "smooth"

if "navigation" in capabilities_list and measurement_layout_used == Measurement_Layout_AAIO_NO_NAVIGATION:
    print("Navigation is in capabilities list, but the measurement layout does not include navigation.")
    raise ValueError
if added_folder != "" and measurement_layout_used == Measurement_Layout_AAIO:
    print("Measurement layout is default, but an added folder is specified.")
    raise ValueError
    
noise_level = np.array([0, 0.4])
capabilities_path = rf"estimated_capabilities/{added_folder}"
N_eval = 200
filename = r"progression_model_results_2M"
incremental_estimator = incremental_measurement_layout(N_eval, capabilities_path, filename=filename, noise_level=noise_level)
incremental_estimator.real_capabilities(measurement_layout_used, capabilities_list)
    
# N_predict = 200
# model_name = r"model_progression/progressionmodel_2M_2000000"
# folder_name = filename
# precise = True

# layout = measurement_layout_used

# i = 0 

# def sorted_alphanumeric(data):
#     convert = lambda text: int(text) if text.isdigit() else text.lower()
#     alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
#     return sorted(data, key=alphanum_key)

# log_dir = "./logs/model_progression"
# files = os.listdir(log_dir)
# files = [f for f in files if "2M" in f]
# files = sorted_alphanumeric(files)
# print("Files in logs directory:", files)

# for model in files:
#     model_name = f"/model_progression/{model[:-4]}" # remove the .zip
#     print(model_name)
#     print(f"Model {model_name} testing for predictive accuracy.")
#     prediction_accuracy(layout, folder_name, added_folder, model_name, N_predict, load_eval = True, precise = precise, noise_level = noise_level, cap_time = i, N_eval = N_eval)
#     i += 1