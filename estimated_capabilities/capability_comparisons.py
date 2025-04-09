import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import json
model_caps_file = "camera_with_frame_stacking_400k"
estimated_navigation_single = np.load(f"estimated_capabilities/{model_caps_file}/ability_navigation_single.npz")["arr_0"]
estimated_visual_single = np.load(f"estimated_capabilities/{model_caps_file}/ability_visual_single.npz")["arr_0"]
estimated_bias_single = np.load(f"estimated_capabilities/{model_caps_file}/ability_bias_rl_single.npz")["arr_0"]
estimated_navigation = np.load(f"estimated_capabilities/{model_caps_file}/navigation_est.npy")
estimated_visual = np.load(f"estimated_capabilities/{model_caps_file}/visual_est.npy")
estimated_bias = np.load(f"estimated_capabilities/{model_caps_file}/bias_est.npy")

print(f"estimated navigation single : {estimated_navigation_single}")
print(f"estimated navigation : {estimated_navigation}")
print(f"estimated visual single : {estimated_visual_single}")
print(f"estimated visual: {estimated_visual}")
print(f"estimated bias single : {estimated_bias_single}")
print(f"estimated bias: {estimated_bias}")