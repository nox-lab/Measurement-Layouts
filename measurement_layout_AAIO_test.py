# %%
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import pandas as pd
from measurement_layout_AAIO import *


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
T = 100  # number of time steps
N = 500  # number of samples
performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
np.random.seed(0)
learn_time_nav = 0.5*T
learn_time_vis = 0.2*T
learn_time_bias = 0.3*T
time_steps = np.linspace(1, T, T)
capability_nav = logistic((time_steps - learn_time_nav)/(T/5))*5.3 #particular point where significant learning occurs, and rate at which this is is determined by the denominator
capability_vis = logistic((time_steps - learn_time_vis)/(T/5))*1.9
capability_bias = logistic((time_steps- 30)/(T/5))
np.random.seed(0)
# Task demand creation, representing a range of arenas
behind = np.random.choice([0, 0.5, 1], N) # 0 if in front of agent's front facing direction, 0.5 if l/r of agent's front facing direction, 1 if behind agent's front facing direction
distance = np.random.uniform(0.1, 5.3, N)
xpos = np.random.choice([-1, 0, 1], N) # -1 if l of agent's actual position, 0 if in line with agent's actual position, 1 if r of agent's actual position
reward_size = np.random.uniform(0.0, 1.9, N)
environmentData["reward_distance"] = distance
environmentData["reward_behind"] = behind
environmentData["abilityMax"] = abilityMax
environmentData["abilityMin"] = abilityMin
environmentData["reward_size"] = reward_size
environmentData["Xpos"] = xpos

fig_demands, ax_demands = plt.subplots()
colors =  ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for i in range(N):
    xpos_current = xpos[i]
    behind_current = behind[i]
    if xpos_current == -1:
        if behind_current == 0:
            color = colors[0]
        elif behind_current == 0.5:
            color = colors[1]
        else:
            color = colors[2]
    elif xpos_current == 0:
        if behind_current == 0:
            color = colors[3]
        elif behind_current == 0.5:
            color = colors[4]
        else:
            color = colors[5]
    else:
        if behind_current == 0:
            color = colors[6]
        elif behind_current == 0.5:
            color = colors[7]
        else:
            color = colors[8]
    ax_demands.plot(distance[i], reward_size[i], marker="o", color=color, alpha=0.2)
ax_demands.set_xlabel("distance")
ax_demands.set_ylabel("reward size")
ax_demands.title.set_text("Demands for arenas")
fig_demands.savefig("demands.png")


rightlefteffect_ = product_on_time_varying(capability_bias, xpos)
perf_nav = logistic(performance_from_capability_and_demand_batch(capability_nav, distance*(behind*0.5+1.0)) + rightlefteffect_)
perf_vis = logistic(performance_from_capability_and_demand_batch(capability_vis, np.log(distance/reward_size)))
successes = np.random.binomial(1, perf_nav*perf_vis, (T, N)) == 1
# Visualise the true values of the data
prop_successes = np.mean(successes, axis=1)
figtest, axtest = plt.subplots()
axtest.bar(range(T), prop_successes, color="grey", label = "successes proportion", alpha=0.2)
axtest.plot(range(T), capability_nav/5.3, label="True capability navigation value")
axtest.plot(range(T), capability_vis/1.9, label="True capability visual value")
axtest.plot(range(T), capability_bias, label="True capability bias value")
plt.xlabel("timestep")
plt.legend()
figtest.savefig("true_values.png")
# %%
m = setupModel(successes, environmentData=environmentData, includeIrrelevantFeatures=includeIrrelevantFeatures, includeNoise=includeNoise, N = N)


with m:
    inference_data = pm.sample(200, target_accept=0.95, cores=2)

bias_fig, bias_ax = plt.subplots()
vis_fig, vis_ax = plt.subplots()
nav_fig, nav_ax = plt.subplots()


for cap, true_mus, fig, ax in [("ability_bias_rl", capability_bias, bias_fig, bias_ax), ("ability_visual", capability_vis, vis_fig, vis_ax), ("ability_navigation", capability_nav, nav_fig, nav_ax)]:
    estimated_p_per_ts = inference_data["posterior"][f"{cap}"].mean(dim=["chain", "draw"])
    # TODO: Understand the hdi function a bit more (why does this 'just work'?)
    estimate_hdis = az.hdi(inference_data["posterior"][f"{cap}"], hdi_prob=0.95)[f"{cap}"]
    low_hdis = [l for l,_ in estimate_hdis]
    high_hdis = [u for _,u in estimate_hdis]
    ax.plot(range(T), true_mus, label=f"True capability {cap} value")
    # TODO: Is it justified to do sigmoid of the mean?
    ax.plot([e for e in estimated_p_per_ts], label="estimated", color="grey")
    # TODO: how does the hdi change after transformation through a sigmoid?
    ax.fill_between([i for i in range(T)], [l for l in low_hdis], [h for h in high_hdis], color="grey", alpha=0.2)
    ax.set_xlabel("timestep")
    ax.legend()
    fig.savefig(f"estimated_{cap}_test.png")