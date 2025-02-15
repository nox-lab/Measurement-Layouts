from measurement_layout_incremental import incremental_measurement_layout
from particles import SMC
from particles.collectors import Moments
import pandas as pd
import os
import numpy as np
import numpy.typing as npt
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION

capabilities_list = ["visual", "bias"]
measurement_layout_used = Measurement_Layout_AAIO_NO_NAVIGATION
directory_in_str = "csv_recordings"
directory = os.fsencode(directory_in_str)
    
for file in os.scandir(directory):
    filename = file.path
    filename = filename.decode()
    if filename.endswith(".csv"):
        print(filename)
    incremental_estimator = incremental_measurement_layout(200, "estimated_capabilities", f"filename")
    incremental_estimator.real_capabilities(measurement_layout_used, capabilities_list)
    continue

    

