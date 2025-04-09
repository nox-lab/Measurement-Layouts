from measurement_layout_incremental import incremental_measurement_layout
from particles import SMC
from particles.collectors import Moments
import pandas as pd
import os
import numpy as np
import numpy.typing as npt
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION

capabilities_list = ["navigation", "visual", "bias"]
measurement_layout_used = Measurement_Layout_AAIO
directory_in_str = "csv_recordings"
capabilities_path = "estimated_capabilities/"
directory = os.fsencode(directory_in_str)
    
for file in os.scandir(directory):
    filename = file.path
    filename = filename.decode()
    if filename.count("\\") > 1:
        continue
    if filename.endswith(".csv"):
        print(filename)
        filename = filename[:-4] # removes the last 4 characters.
        print(filename)
        filename = filename.split("\\")[-1] # Removes the parent folder.
        print(filename)
    else:
        continue
    if not os.path.exists(f"{capabilities_path}/{filename}"):
        print(filename)
        incremental_estimator = incremental_measurement_layout(200, capabilities_path, filename)
        incremental_estimator.real_capabilities(measurement_layout_used, capabilities_list)

# measurement_layout_used = Measurement_Layout_AAIO
# incremental_estimator = incremental_measurement_layout(500, "estimated_capabilities", filename="", testing= True, noise_level = np.array([0.0, 0.3]))
# incremental_estimator.base_test(measurement_layout_used)

    

