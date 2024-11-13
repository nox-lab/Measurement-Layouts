# %%
import pymc as pm
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import arviz as az
from typing import Callable

# %% [markdown]
# # Hyperparameters

# %%
T = 100  # number of time steps
N = 100  # number of samples

# %% [markdown]
# # Synthetic Data Generatation
if __name__ == "__main__":
    # %%
    # Helper to transform capability and demand into a performance probability
    performance_from_capability_and_demand_batch: Callable[[npt.ArrayLike,npt.ArrayLike], npt.ArrayLike] = lambda capability, demand : (capability[:,None]-demand + 1 ) / 2

    # Generate some example data
    np.random.seed(0)
    # Dimension: (T,)
    true_mu_a = (np.sin(np.linspace(0, 3*np.pi, T)) + 1) / 2  # sin transformed to be in [0,1]
    true_mu_b = np.linspace(0, 1, T)  # Linearly increasing capability
    true_mu_c = (np.sin(np.linspace(np.pi, (5/2)*np.pi, T)) + 1) / 2  # Dropping then rising capability
    # Dimension: (T, N)
    instance_demands_a = np.random.uniform(0,1, (T,N))
    instance_demands_b = np.random.uniform(0,1, (T,N))
    instance_demands_c = np.random.uniform(0,1, (T,N))
    # Dimension: (T,N)
    perf_a = performance_from_capability_and_demand_batch(true_mu_a, instance_demands_a)
    perf_b = performance_from_capability_and_demand_batch(true_mu_b, instance_demands_b)
    perf_c = performance_from_capability_and_demand_batch(true_mu_c, instance_demands_c)
    # successes = (np.random.binomial(1, performance_from_capability_and_demand_batch(true_mu, instance_demands), (T,N)) == 1)
    successes = np.random.binomial(1, perf_a*perf_b*perf_c, (T,N)) == 1

    # %%
    # Visualise the true values of the data
    prop_successes = np.mean(successes, axis=1)
    plt.bar(range(T), prop_successes, color="grey", alpha=0.2)
    for true_cap in [true_mu_a,true_mu_b,true_mu_c]:
        plt.plot(range(T), true_cap, label="Capability value")
    plt.xlabel("timestep")
    plt.legend()
    plt.show()

    # %% [markdown]
    # # Measurement Layout

    # %%
    sigmoid: Callable[[float],float] = lambda x : (1 + np.e**(-x))**(-1)

    # Define the model
    m = pm.Model()

    with m:
        # Demands
        demands_a = pm.MutableData("demands_a", instance_demands_a)
        demands_b = pm.MutableData("demands_b", instance_demands_b)
        demands_c = pm.MutableData("demands_c", instance_demands_c)

        # Priors
        sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma=1.0)
        sigma_c = pm.HalfNormal("sigma_c", sigma=1.0)
        
        # Hidden state
        cap_a_raw = pm.GaussianRandomWalk("cap_a_raw", sigma=sigma_a, shape=T)
        cap_b_raw = pm.GaussianRandomWalk("cap_b_raw", sigma=sigma_b, shape=T)
        cap_c_raw = pm.GaussianRandomWalk("cap_c_raw", sigma=sigma_c, shape=T)

        # Transform p to be between 0 and 1 using a sigmoid function
        cap_a = pm.Deterministic("cap_a", sigmoid(cap_a_raw))
        cap_b = pm.Deterministic("cap_b", sigmoid(cap_b_raw))
        cap_c = pm.Deterministic("cap_c", sigmoid(cap_c_raw))
        
        # Observations
        performance_a = pm.Deterministic("performanceA", performance_from_capability_and_demand_batch(cap_a, demands_a))
        performance_b = pm.Deterministic("performanceB", performance_from_capability_and_demand_batch(cap_b, demands_b))
        performance_c = pm.Deterministic("performanceC", performance_from_capability_and_demand_batch(cap_c, demands_c))
        observed = pm.Bernoulli("ObservedPerformance", performance_a*performance_b*performance_c, observed=successes)

    # %%
    # Inference
    with m:
        inference_data = pm.sample(1000, target_accept=0.95, cores=10)

    # %% [markdown]
    # # Results analysis

    # %%
    for cap, true_mus in [("a", true_mu_a), ("b", true_mu_b), ("c", true_mu_c)]:
        estimated_p_per_ts = inference_data["posterior"][f"cap_{cap}_raw"].mean(dim=["chain", "draw"])
        # TODO: Understand the hdi function a bit more (why does this 'just work'?)
        estimate_hdis = az.hdi(inference_data["posterior"][f"cap_{cap}_raw"], hdi_prob=0.95)[f"cap_{cap}_raw"]
        low_hdis = [l for l,_ in estimate_hdis]
        high_hdis = [u for _,u in estimate_hdis]
        plt.plot(range(T), true_mus, label=f"True capability {cap} value")
        # TODO: Is it justified to do sigmoid of the mean?
        plt.plot([sigmoid(e) for e in estimated_p_per_ts], label="estimated", color="grey")
        # TODO: how does the hdi change after transformation through a sigmoid?
        plt.fill_between([i for i in range(T)], [sigmoid(l) for l in low_hdis], [sigmoid(h) for h in high_hdis], color="grey", alpha=0.2)
        plt.xlabel("timestep")
        plt.legend()
        plt.show()

    # %%
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Visualise the true values of the data
    for cap, true_mus in [("a", true_mu_a), ("b", true_mu_b), ("c", true_mu_c)]:
        ax1.plot(range(T), true_mus)
        estimated_p_per_ts = inference_data["posterior"][f"cap_{cap}_raw"].mean(dim=["chain", "draw"])
        # TODO: Understand the hdi function a bit more (why does this 'just work'?)
        estimate_hdis = az.hdi(inference_data["posterior"][f"cap_{cap}_raw"], hdi_prob=0.95)[f"cap_{cap}_raw"]
        low_hdis = [l for l,_ in estimate_hdis]
        high_hdis = [u for _,u in estimate_hdis]

        ax2.plot([sigmoid(e) for e in estimated_p_per_ts], label="estimated")


