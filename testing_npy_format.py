import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import pandas as pd

mean_cap = np.load("estimated_ability_navigation_test.png_based_on_NOTHING.npy")
high_hdi = np.load("estimated_ability_navigation_test.png_based_on_NOTHING_high_hdi.npy")
low_hdi = np.load("estimated_ability_navigation_test.png_based_on_NOTHING_low_hdi.npy")
print(mean_cap)
print(high_hdi)
print(low_hdi)