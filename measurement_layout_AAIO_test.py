# %%
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import pandas as pd
from measurement_layout_AAIO import *

if __name__ == "__main__":
    includeIrrelevantFeatures = True
    includeNoise=False
    test_synthetic = False
    save_samples = False
    show_max = False
    environmentData = dict()
    abilityMax = {
        "navigationAbility": 5.3,
        "visualAbility": 10,
    }
    abilityMin = {
        "navigationAbility": 0.0,
        "visualAbility": -999,
    }

    # FIGURES TO CREATE ON WHICH WE PUT CAPABILITIES
    def logit(x):
        return np.log(x/(1-x))
    
    performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand)
    product_on_time_varying: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]*demand)
    environmentData["abilityMax"] = abilityMax
    environmentData["abilityMin"] = abilityMin
    all_capabilities = ["ability_navigation", "ability_visual", "ability_bias_rl"]
    excluded_capabilities_string = ""
    maximum_capabilites = None
    if test_synthetic:
        filename_no_ext = "NOTHING"
        filename_no_ext_or_pref = "NOTHING"
        T = 25  # number of time steps
        N = 10  # number of samples
        learn_time_nav = 0.5*T
        learn_time_vis = 0.2*T
        learn_time_bias = 0.3*T
        time_steps = np.linspace(1, T, T)
        capability_nav = logistic((time_steps - learn_time_nav)/(T/5))*5.3 #particular point where significant learning occurs, and rate at which this is is determined by the denominator
        capability_vis = logistic((time_steps - learn_time_vis)/(T/5))*1.9
        capability_bias = logistic((time_steps- 30)/(T/5))
        np.random.seed(0)
        included_capabilities = [["ability_navigation", capability_nav], ["ability_visual", capability_vis], ["ability_bias_rl", capability_bias]]
        relevant_figs = [(cap, plt.subplots()) for cap in included_capabilities]
        # Task demand creation, representing a range of arenas
        behind = np.random.choice([0, 0.5, 1], N) # 0 if in front of agent's front facing direction, 0.5 if l/r of agent's front facing direction, 1 if behind agent's front facing direction
        distance = np.random.uniform(0.1, 5.3, N)
        xpos = np.random.choice([-1, 0, 1], N) # -1 if l of agent's actual position, 0 if in line with agent's actual position, 1 if r of agent's actual position
        reward_size = np.random.uniform(0.0, 1.9, N)
        environmentData["reward_distance"] = distance
        environmentData["reward_behind"] = behind
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
        axtest.set_xlabel("timestep")
        axtest.legend()
        figtest.savefig("true_values.png")
        # Visualise the true values of the capability profiles
        for cap, (fig, ax) in relevant_figs:
            ax.plot(range(T), cap[1], label=f"True capability {cap[0]} value")
    else:
        folder = "csv_recordings"
        filename_no_ext_or_pref = "raycasts_with_frame_stacking_500k" # WRITE THE NAME OF THE FILE HERE
        filename_no_ext = folder + r"/" + filename_no_ext_or_pref
        filename = filename_no_ext + ".csv"
        # filename = "fixed_hopefully_test_file.csv" 
        N = 200  # number of arenas
        excluded_capabilities = []
        excluded_capabilities_string = "_".join(excluded_capabilities)
        included_capabilities = [c for c in all_capabilities if c not in excluded_capabilities]
        df_final = pd.read_csv("./" + filename)
        maximum_distance = df_final["reward_distance"].max()
        minimum_size = df_final["reward_size"].min()
        nav_max = maximum_distance*1.5 + logit(0.99) # associated to 0.99 nav performance
        vis_max = logit(0.99) + np.log(maximum_distance) - np.log(minimum_size) # associated to 0.99 vis performance
        bias_max = 2*logit(0.99) # associated to swapping of success rate from 0.99 to 0.01
        maximum_capabilities = {
            "ability_navigation": [nav_max],
            "ability_visual": [vis_max],
            "ability_bias_rl": [bias_max, -bias_max]
        }
        successes = df_final["reward"].values # Current in NT form
        successes = successes.reshape((-1, N)) # We want T x N
        successes = successes > -0.9
        T = successes.shape[0]  # number of time steps
        environmentData["reward_distance"] = df_final["reward_distance"].values[0:N]
        environmentData["reward_behind"] = df_final["reward_behind"].values[0:N]
        environmentData["reward_size"] = df_final["reward_size"].values[0:N]
        environmentData["Xpos"] = df_final["Xpos"].values[0:N]
        total_successes = np.mean(successes, axis=1)
        fig, ax = plt.subplots()
        ax.bar(range(T), total_successes, color="grey", alpha=0.2)
        ax.set_xlabel("timestep")
        ax.set_ylabel("proportion of successes")
        ax.title.set_text("Successes over time")
        fig.savefig(f"./estimated_capabilities/{filename_no_ext_or_pref}/successes.png")
        relevant_figs = [([cap], plt.subplots()) for cap in included_capabilities]

    # %%
    m = setupModel(successes, environmentData=environmentData, includeIrrelevantFeatures=includeIrrelevantFeatures, includeNoise=includeNoise, N = N)


    with m:
        inference_data = pm.sample(1000, target_accept=0.95, cores=2)

    if test_synthetic:
        final_str = "_test.png"
    else:
        if save_samples:    
            az.to_netcdf(inference_data, f"inference_data_{filename_no_ext}.nc")
        if excluded_capabilities_string == "":
            final_str = ".png"
        else:
            final_str = "_excluding_" + excluded_capabilities_string + ".png"
    for cap, (fig, ax) in relevant_figs:
        cap = cap[0]
        estimated_p_per_ts = inference_data["posterior"][f"{cap}"].mean(dim=["chain", "draw"])
        
        # Save the estimated capabilities
        np.save(f"./estimated_capabilities/{filename_no_ext_or_pref}/_estimated_{cap}_FULL.npy", estimated_p_per_ts)
        # TODO: Understand the hdi function a bit more (why does this 'just work'?)
        estimate_hdis = az.hdi(inference_data["posterior"][f"{cap}"], hdi_prob=0.95)[f"{cap}"]
        low_hdis = [l for l,_ in estimate_hdis]
        high_hdis = [u for _,u in estimate_hdis]
        np.save(f"./estimated_capabilities/{filename_no_ext_or_pref}/_estimated_{cap}_FULL_low_hdi.npy", np.array(low_hdis))
        np.save(f"./estimated_capabilities/{filename_no_ext_or_pref}/_estimated_{cap}_FULL_high_hdi.npy", np.array(high_hdis))
        # TODO: Is it justified to do sigmoid of the mean?
        ax.plot([e for e in estimated_p_per_ts], label="Mean Estimate", color="grey")
        # TODO: how does the hdi change after transformation through a sigmoid?
        ax.fill_between([i for i in range(T)], [l for l in low_hdis], [h for h in high_hdis], color="grey", label = "95% HDI", alpha=0.2)
        if show_max and maximum_capabilites is not None:
            for maximum_cap in maximum_capabilities[cap]:
                ax.axhline(maximum_cap, color="red", linestyle="--")
            ax.plot([], [], color="red", linestyle="--", label="Capability bound")
        ax.set_title(f"Estimated {cap}")
        ax.set_xlabel("Optimisation Steps")
        ax.legend()
        fig.savefig(f"./estimated_capabilities/{filename_no_ext_or_pref}/{cap}_profile_FULL.png")
