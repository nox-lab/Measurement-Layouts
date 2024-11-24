import pandas as pd  
import numpy as np

df_final = pd.read_csv("evaluation_results_with_new_train.csv")
successes = df_final["reward"].values # Current in NT form
successes = successes.reshape((-1, 20)) # We want T x N
print(successes.shape)