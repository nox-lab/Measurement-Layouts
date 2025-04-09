import measurement_layout_incremental
from various_measurement_layouts import Measurement_Layout_AAIO
import os

ML_AAIO_inc = measurement_layout_incremental.incremental_measurement_layout(100, "estimated_capabilities", "testing_incremental", testing = True)


ML_AAIO_inc.base_test(Measurement_Layout_AAIO)