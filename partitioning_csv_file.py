import pandas as pd
import numpy as np
# Convert to DataFrame
import matplotlib.pyplot as plt
import json
from io import StringIO

def partition(input_file_path, partitions: list, N, relevant_timestep):

    df = pd.read_csv(input_file_path)
    start_row = N * relevant_timestep
    end_row = N * (relevant_timestep + 1)
    df = df.iloc[start_row:end_row]
    df["successes"] = (df["reward"] > -0.9).astype(int)
    # gonna add columns for the partitions
    partition_labels = []
    for partition in partitions:
        if partitions[partition] == -1:
            partition_labels.append(partition)
            continue
        df[f"{partition}_bin"] = pd.cut(df[partition], bins = partitions[partition])
        partition_labels.append(f"{partition}_bin")
        
   

    # Partition data by Xpos and reward_behind
    partitions = df.groupby(partition_labels).agg(['mean'])

    # Display the partitioned data
    print(partitions.head(30))
    return partitions
    
def correlations(input_file_path, correlates: list, N, relevant_timestep):
    df = pd.read_csv(input_file_path)
    start_row = N * relevant_timestep
    end_row = N * (relevant_timestep + 1)
    df = df.iloc[start_row:end_row]
    df["successes"] = (df["reward"] > -0.9).astype(int)
    
    return df[correlates].corr("pearson", df["successes"])
    
    
def subsampled_evaluations(input_file, N_new, N):
    # We wish to take, for instance, a 10000 rows and select N_new rows / 200 for each 200, so 
    # 1, 201, 401 ... and maybe 50, 250 ... etc.
    df = pd.read_csv(input_file)
    number_of_rows = len(df)
    all_indices = np.arange(N_new)
    relevant_indices = np.mgrid[0:N_new, 0:number_of_rows:N].sum(axis=0)
    relevant_indices=  relevant_indices.flatten("F")
    new_df = df.iloc[relevant_indices]
    new_df.to_csv(f"{input_file[:-4]}_subsampled_{N_new}.csv", index = False)
N = 200
relevant_timesteps = [i for i in range(40)]
partition_headers_and_ranges = {
    "reward_behind": 3,
}

pd.set_option("display.max_rows", 15)

fig, ax = plt.subplots(figsize=(10, 5))
all_negative_xpos = []
all_positive_xpos = []
csv_file = "working_caps_predictive_3" # NO EXTENSION
for relevant_timestep in relevant_timesteps:
    partitioned_data = partition(f"csv_recordings/{csv_file}.csv", partition_headers_and_ranges, N, relevant_timestep)
    all_negative_xpos.append(partitioned_data["successes"].iloc[2]) # behind
    all_positive_xpos.append(partitioned_data["successes"].iloc[1]) # to the side
    print(partitioned_data["successes"].iloc[2].to_numpy(), partitioned_data["successes"].iloc[1].to_numpy(), relevant_timestep)
    # print(input("continue"))
    
all_negative_xpos = np.array(all_negative_xpos)
all_positive_xpos = np.array(all_positive_xpos)
ratios_array = all_positive_xpos-all_negative_xpos
ratios_array=  ratios_array.squeeze()
print(ratios_array)

# ax.plot(relevant_timesteps, all_positive_xpos, marker = "s", linestyle= "-", label = "Xpos < 0")
# ax.plot(relevant_timesteps, all_negative_xpos, marker = "o", linestyle= "-", label = "Xpos > 0")
ax.plot(relevant_timesteps, ratios_array, marker = "o", linestyle= "-", label = "Unmodified Arenas")



all_negative_xpos = []
all_positive_xpos = []
csv_file = "progression_model_results_2M_N1000" # NO EXTENSION
for relevant_timestep in relevant_timesteps:
    partitioned_data = partition(f"csv_recordings/{csv_file}.csv", partition_headers_and_ranges, N, relevant_timestep)
    all_negative_xpos.append(partitioned_data["successes"].iloc[2]) # behind
    all_positive_xpos.append(partitioned_data["successes"].iloc[1]) # to the side
    print(partitioned_data["successes"].iloc[2].to_numpy(), partitioned_data["successes"].iloc[1].to_numpy(), relevant_timestep)
    # print(input("continue"))
    
all_negative_xpos = np.array(all_negative_xpos)
all_positive_xpos = np.array(all_positive_xpos)
ratios_array = all_positive_xpos-all_negative_xpos
ratios_array=  ratios_array.squeeze()
print(ratios_array)

ax.plot(relevant_timesteps, ratios_array, marker = "o", linestyle= "-", label = "Modified Arenas")
# ax.set_title(r"$\hat{y}_{D_{behind} = 0.5}$ - $\hat{y}_{D_{behind} = 1}$")
ax.set_title("Extra Success From Reward Movement From Behind to Side")
ax.set_xlabel("Optimisation Steps")
ax.set_ylabel("Proportion of Successes Difference")
ax.legend()
ax.grid()
plt.savefig(f"Xpos_successes_for_{csv_file}")
plt.show()
    
    