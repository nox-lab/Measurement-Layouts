from measurement_layout_incremental import incremental_measurement_layout
import pandas as pd
import os
import numpy as np
import numpy.typing as npt
from predictive_accuracy import prediction_accuracy
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION, Measurement_Layout_AAIO_precise

capabilities_list = ["visual", "bias", "noise"]
measurement_layout_used = Measurement_Layout_AAIO_NO_NAVIGATION
noise_level = 0.0
directory_in_str = "csv_recordings"
capabilities_path = "estimated_capabilities/no_navigation"
filename = r"progression_model_results_2M_N1000"
incremental_estimator = incremental_measurement_layout(1000, capabilities_path, filename=filename, noise_level=noise_level)
incremental_estimator.real_capabilities(measurement_layout_used, capabilities_list)
    
N = 200
model_name = "model_progression/progressionmodel_2M_2000000"
folder_name = "progression_model_results_2M_N1000"
added_folder = "no_navigation"
precise = True

layout = measurement_layout_used
prediction_accuracy(layout, folder_name, added_folder, model_name, N, load_eval = True, precise = precise, noise_level = noise_level)
