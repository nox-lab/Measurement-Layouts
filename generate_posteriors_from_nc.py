import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt

filename = ""
caps = ["ability_navigation", "ability_visual", "ability_bias_rl"]
relevant_figs = [([cap], plt.subplots()) for cap in caps]   
inference_data = az.from_netcdf("inference_data.nc")
final_str = ""
    
for cap, (fig, ax) in relevant_figs:
    cap = cap[0]
    estimated_p_per_ts = inference_data["posterior"][f"{cap}"].mean(dim=["chain", "draw"])
    # TODO: Understand the hdi function a bit more (why does this 'just work'?)
    estimate_hdis = az.hdi(inference_data["posterior"][f"{cap}"], hdi_prob=0.95)[f"{cap}"]
    low_hdis = [l for l,_ in estimate_hdis]
    high_hdis = [u for _,u in estimate_hdis]
    # TODO: Is it justified to do sigmoid of the mean?
    ax.plot([e for e in estimated_p_per_ts], label="estimated", color="grey")
    # TODO: how does the hdi change after transformation through a sigmoid?
    ax.fill_between([i for i in range(T)], [l for l in low_hdis], [h for h in high_hdis], color="grey", alpha=0.2)
    ax.set_title(f"Estimated {cap}")
    ax.set_xlabel("timestep")
    ax.legend()
    fig.savefig(f"estimated_{cap}{final_str}_based_on_{filename}")