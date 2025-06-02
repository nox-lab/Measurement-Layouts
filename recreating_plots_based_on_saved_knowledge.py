import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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


fig, ax = plt.subplots()

ax.plot(mean_cap, label='Mean Capability', color='blue')
ax.fill_between(range(len(mean_cap)), low_hdi, high_hdi, color='blue', alpha=0.2, label='95% HDI')
ax.set_title('Estimated Navigation Capability')
ax.set_xlabel('Optimisation Step')
ax.set_ylabel('Capability')