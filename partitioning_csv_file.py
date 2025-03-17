import pandas as pd
import numpy as np
# Convert to DataFrame
from io import StringIO

def partition(input_file_path, partitions: list, N, relevant_timestep):

    df = pd.read_csv(input_file_path)
    start_row = N * relevant_timestep
    end_row = N * (relevant_timestep + 1)
    df = df.iloc[start_row:end_row]
    df["successes"] = (df["reward"] > 0.9).astype(int)
    # gonna add columns for the partitions
    partition_labels = []
    for partition in partitions:
        if partitions[partition] == -1:
            partition_labels.append(partition)
            continue
        df[f"{partition}_bin"] = pd.cut(df[partition], bins = partitions[partition])
        partition_labels.append(f"{partition}_bin")
        
   

    # Partition data by Xpos and reward_behind
    partitions = df.groupby(partition_labels).agg(['mean', 'std'])

    # Display the partitioned data
    print(partitions.head(15))
    
def correlations(input_file_path, correlates: list, N, relevant_timestep):
    df = pd.read_csv(input_file_path)
    start_row = N * relevant_timestep
    end_row = N * (relevant_timestep + 1)
    df = df.iloc[start_row:end_row]
    df["successes"] = (df["reward"] > 0.9).astype(int)
    
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
N = 1000
relevant_timesteps = [5]
partition_headers_and_ranges = {
    "Xpos" : -1,
    "reward_behind": 4,
}

pd.set_option("display.max_rows", 15)
for relevant_timestep in relevant_timesteps:
    partition("csv_recordings/progression_model_results_2M_N1000.csv", partition_headers_and_ranges, N, relevant_timestep)