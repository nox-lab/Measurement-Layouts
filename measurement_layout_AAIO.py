# %%
import pymc as pm
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import arviz as az
from typing import Callable

def logistic(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    m = pm.Model()
    T = 100  # number of time steps
    N = 100  # number of samples/number of arenas.
    performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
    np.random.seed(0)
    learn_time_nav = 50
    learn_time_vis = 20
    learn_time_bias = 30
    time_steps = np.linspace(1, T, T)
    capability_nav = logistic((time_steps - learn_time_nav)/20)*5.3 #particular point where significant learning occurs, and rate at which this is is determined by the denominator
    capability_vis = logistic((time_steps - learn_time_vis)/10)*1.9
    capability_bias = 1/time_steps

    plt.plot(time_steps,capability_nav/5.3, label = "navigation capability normalised")
    plt.plot(time_steps,capability_vis/1.9, label = "visual capability normalised")
    plt.plot(time_steps,capability_bias, label = "left-right bias")
    # Task capability creation, representing a range of arenas
    behind = np.random.choice([0, 0.5, 1], N) # 0 if in front of agent's front facing direction, 0.5 if l/r of agent's front facing direction, 1 if behind agent's front facing direction
    distance = np.random.uniform(0, 5.3, N)
    xpos = np.random.choice([-1, 0, 1]) # -1 if l of agent's actual position, 0 if in line with agent's actual position, 1 if r of agent's actual position
    reward_size = np.random.uniform(0, 1.9, N)
    rightlefteffect_ = capability_bias*xpos
    perf_nav = logistic(performance_from_capability_and_demand_batch(capability_nav + rightlefteffect_, distance*(1/2 * behind + 1)))
    perf_vis = logistic(performance_from_capability_and_demand_batch(capability_vis, reward_size))
    print(np.min(perf_nav), np.max(perf_nav))
    print(np.min(perf_vis), np.max(perf_vis))
    print(perf_vis.shape)
    print(perf_nav.shape)
    successes = np.random.binomial(1, perf_nav*perf_vis, (T, N)) == 1

    # Visualise the true values of the data
    prop_successes = np.mean(successes, axis=1)
    plt.bar(range(T), prop_successes, color="grey", alpha=0.2)
    plt.xlabel("timestep")
    plt.legend()
    plt.show()
  #%%
    with m:
        demands_distance = pm.MutableData("demands_distance", behind)
        demands_behind = pm.MutableData("demands_behind", distance)
        demands_size = pm.MutableData("demands_size", reward_size)
        demands_xpos = pm.MutableData("demands_xpos", xpos)
        
        # Priors
        sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma=1.0)
        sigma_performance = pm.Uniform("sigma_performance", lower=0, upper=1)
        
        ability_nav_raw = pm.GaussianRandomWalk("ability_navigation", mu=0, sigma=sigma_a, shape = T)
        ability_visual_raw = pm.GaussianRandomWalk("ability_visual", mu=0, sigma=sigma_b, shape = T)
        ability_bias_rl_raw = pm.GaussianRandomWalk("ability_bias_rl", mu=0, sigma=1, shape = T)
        
        ability_nav = pm.Deterministic("cap_a", ability_nav_raw)
        ability_visual = pm.Deterministic("cap_b", ability_visual_raw) # moved to (0,1) so to satisfy mentioned dist change.
        ability_bias_rl = pm.Deterministic("cap_c", ability_bias_rl_raw/np.sqrt(T)) # resultant bias is N(0, 1)
        
        rightlefteffect = pm.Deterministic("rightleftbias", ability_bias_rl*demands_xpos)
        
        navigation_performance = pm.Deterministic("navigation_performance", logistic(performance_from_capability_and_demand_batch(ability_nav + rightlefteffect, demands_distance*(1/2 * demands_behind + 1))))
        visual_performance = pm.Deterministic("visual_performance", logistic(performance_from_capability_and_demand_batch(ability_visual, demands_size)))
        task_performance = pm.Bernoulli("task_performance", p=(1 - sigma_performance)*(navigation_performance*visual_performance) + sigma_performance*1, observed=successes)
        
    with m:
        inference_data = pm.sample(1000, target_accept=0.95, cores=2)
        

    for cap, true_mus in [("navigation", capability_nav), ("visual", capability_vis), ("bias_rl", capability_bias)]:
        estimated_p_per_ts = inference_data["posterior"][f"ability_{cap}"].mean(dim=["chain", "draw"])
        # TODO: Understand the hdi function a bit more (why does this 'just work'?)
        estimate_hdis = az.hdi(inference_data["posterior"][f"ability_{cap}"], hdi_prob=0.95)[f"ability_{cap}"]
        low_hdis = [l for l,_ in estimate_hdis]
        high_hdis = [u for _,u in estimate_hdis]
        plt.plot(range(T), true_mus, label=f"True capability {cap} value")
        # TODO: Is it justified to do sigmoid of the mean?
        plt.plot([logistic(e) for e in estimated_p_per_ts], label="estimated", color="grey")
        # TODO: how does the hdi change after transformation through a sigmoid?
        plt.fill_between([i for i in range(T)], [logistic(l) for l in low_hdis], [logistic(h) for h in high_hdis], color="grey", alpha=0.2)
        plt.xlabel("timestep")
        plt.legend()
        plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Visualise the true values of the data
    for cap, true_mus in [("navigation", capability_nav), ("visual", capability_vis), ("bias_rl", capability_bias)]:
        ax1.plot(range(T), true_mus)
        estimated_p_per_ts = inference_data["posterior"][f"ability_{cap}"].mean(dim=["chain", "draw"])
        # TODO: Understand the hdi function a bit more (why does this 'just work'?)
        estimate_hdis = az.hdi(inference_data["posterior"][f"ability_{cap}"], hdi_prob=0.95)[f"ability_{cap}"]
        low_hdis = [l for l,_ in estimate_hdis]
        high_hdis = [u for _,u in estimate_hdis]

        ax2.plot([logistic(e) for e in estimated_p_per_ts], label="estimated")
            
        