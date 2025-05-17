from gym.spaces import Box
import numpy as np
import pandas as pd
import os
import re

def is_image_space(space: Box, check_channels_last: bool = True) -> bool:
    """
    Check if a Box space is an image space.
    It must have dtype uint8 and shape (H, W, C) if channels_last, else (C, H, W).
    """
    if not isinstance(space, Box) or space.dtype != np.uint8:
        return False

    if check_channels_last:
        return len(space.shape) == 3 and space.shape[2] in [1, 3]
    else:
        return len(space.shape) == 3 and space.shape[0] in [1, 3]


def combine_csv_files(input_files : list , N_values : list, output_file : str = "combined_data.csv"):
    """
    Combine multiple CSV files into one, ensuring that the number of rows in each file is equal to the corresponding N value.
    """
    combined_data = []
    for input_file, N in zip(input_files, N_values):
        df = pd.read_csv(input_file)
        if len(df) != N:
            raise ValueError(f"File {input_file} does not have {N} rows.")
        combined_data.append(df)

    combined_df = pd.concat(combined_data, ignore_index=True)
    return combined_df


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

if __name__ == "__main__":
    initial_file_main = "csv_recordings/progression_model_results_400k_camera.csv"
    initial_file_main_edited = initial_file_main[:-4] + "_edited.csv"
    model_files = [os.path.join(dirpath,f) for (dirpath, dirnames, filenames) in os.walk("csv_recordings/predictive_data/model_progression_camera/") for f in filenames]
    print(model_files)
    model_files = sorted_alphanumeric(model_files)
    N = 100
    N_new = 200


    for time_step in range(40):
        model_file_name = model_files[time_step]
        print(model_file_name, len(model_files), time_step)
        df_model = pd.read_csv(model_file_name)
        df_model_remaining  = df_model.iloc[:N_new-N]
        df_model = df_model.iloc[N_new-N:]
        # print(df_model)
        # print(df_model_remaining)
        print(input("continue"))
        df_initial = pd.read_csv(initial_file_main)
        insertion_index = time_step * N_new
        if insertion_index > len(df_initial):
            raise ValueError(f"Insertion index {insertion_index} exceeds the number of rows in the initial file.")
        df_initial = pd.concat([df_initial.iloc[:insertion_index], df_model, df_initial.iloc[insertion_index:]], ignore_index=True)
        df_initial.to_csv(initial_file_main, index=False)
        print(df_model)
        df_model_remaining.to_csv(model_file_name, index = False)
        
        
    