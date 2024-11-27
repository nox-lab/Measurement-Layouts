# %%
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import pandas as pd
from measurement_layout_AAIO import setupModel


includeIrrelevantFeatures = True
includeNoise=True
environmentData = dict()
abilityMax = {
    "navigationAbility": 5.3,
    "visualAbility": 1.9,
}
abilityMin = {
    "navigationAbility": 0.0,
    "visualAbility": -999,
}
def rbf_kernel(X, length_scale=1.0):
    """
    Generates a covariance matrix using the RBF (squared exponential) kernel.

    Parameters:
    X (numpy.ndarray): The input data, a 1D array of points.
    length_scale (float): The length scale parameter for the RBF kernel.

    Returns:
    numpy.ndarray: The covariance matrix.
    """
    # Calculate the pairwise squared Euclidean distances
    sq_dists = np.subtract.outer(X, X) ** 2

    # Compute the covariance matrix using the RBF kernel
    K = np.exp(-sq_dists / (2 * length_scale ** 2))

    return K


def logistic(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
  
 
  N = 200  # number of samples
  performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
  product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
  np.random.seed(0)
  df_final = pd.read_csv("evaluation_results_with_new_train.csv")
  successes = df_final["reward"].values # Current in NT form
  successes = successes.reshape((-1, N)) # We want T x N
  T = successes.shape[0]  # number of time steps
  environmentData["reward_distance"] = df_final["reward_distance"].values[0:N]
  environmentData["reward_behind"] = df_final["reward_behind"].values[0:N]
  environmentData["reward_size"] = df_final["reward_size"].values[0:N]
  environmentData["Xpos"] = df_final["Xpos"].values[0:N]
  successes[successes > -1] = 1
  successes[successes <= -1] = 0
  print(successes)
# %%
  m = setupModel(successes, cholesky=None, environmentData=environmentData, includeIrrelevantFeatures=includeIrrelevantFeatures, includeNoise=includeNoise, N = N)
  
    
  with m:
    inference_data = pm.sample(500, target_accept=0.95, cores=2)
    
  bias_fig, bias_ax = plt.subplots()
  vis_fig, vis_ax = plt.subplots()
  nav_fig, nav_ax = plt.subplots()
  

  for cap, fig, ax in [("ability_bias_rl", bias_fig, bias_ax), ("ability_visual", vis_fig, vis_ax), ("ability_navigation", nav_fig, nav_ax)]:
      estimated_p_per_ts = inference_data["posterior"][f"{cap}"].mean(dim=["chain", "draw"])
      # TODO: Understand the hdi function a bit more (why does this 'just work'?)
      estimate_hdis = az.hdi(inference_data["posterior"][f"{cap}"], hdi_prob=0.95)[f"{cap}"]
      low_hdis = [l for l,_ in estimate_hdis]
      high_hdis = [u for _,u in estimate_hdis]
      # TODO: Is it justified to do sigmoid of the mean?
      ax.plot([e for e in estimated_p_per_ts], label="estimated", color="grey")
      # TODO: how does the hdi change after transformation through a sigmoid?
      ax.fill_between([i for i in range(T)], [l for l in low_hdis], [h for h in high_hdis], color="grey", alpha=0.2)
      plt.xlabel("timestep")
      plt.legend()
      fig.savefig(f"estimated_{cap}.png")
