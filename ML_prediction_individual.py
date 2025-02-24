from measurement_layout_incremental import incremental_measurement_layout
import pandas as pd
import os
import numpy as np
import numpy.typing as npt
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION, Measurement_Layout_AAIO_precise

capabilities_list = ["visual", "bias"]
measurement_layout_used = Measurement_Layout_AAIO_NO_NAVIGATION
directory_in_str = "csv_recordings"
capabilities_path = "estimated_capabilities/no_navigation"
filename = r"progression_model_results_2M"
incremental_estimator = incremental_measurement_layout(200, capabilities_path, filename=filename)
incremental_estimator.real_capabilities(measurement_layout_used, capabilities_list)
    

