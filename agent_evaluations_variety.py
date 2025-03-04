from stable_baselines3 import PPO
from stable_baselines3 import A2C
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
import torch as th
from ChangeEnvAfterEpisode import EndofEpisodeReward
from evaluation_reward_ep_wrapper import EvalRewardCallback
from generating_configs import Demands
import sys
import random
import pandas as pd
import os
import numpy as np
from animal_ai_reset_wrapper import AnimalAIReset

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.environment import AnimalAIEnvironment
from generating_configs import gen_config_from_demands_batch_random, gen_config_from_demands_batch
from generating_configs_class import ConfigGenerator
import subprocess
from test_animal_ai import train_agent_configs
import re


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)



if __name__ == "__main__":
    N = 1000
    N_train = 400
    config_generator = ConfigGenerator(very_precise = True)
    # Currently we have been using random demands, but we can make them SPECIAL
    demands_list_train = []
    dumb_string = config_generator.gen_config_from_demands_batch_random(N_train, filename=r"example_batch_train.yaml", time_limit=200, dist_max = 8, size_min = 0.5, size_max = 6, numbered = False, seed = 101)
    dumb_string_2, demands_list = config_generator.gen_config_from_demands_batch_random(N, r"example_batch_eval.yaml", time_limit=150, dist_max=15, size_min = 0.5, size_max = 6, numbered = False, seed = 1)
    eval_freq = 1
    env_path_train = r"..\WINDOWS\AAI\Animal-AI.exe"
    env_path_eval = r"..\WINDOWS\AAI - Copy\Animal-AI.exe"
    configuration_file_train = r"example_batch_train.yaml"  # !!!!! ODD NUMBER OF ARENAS REQUIRED skips arenas for some reason !!!!!
    configuration_file_eval = r"example_batch_eval.yaml"
    
    recording_file = r"./progression_model_results_2M_N1000.csv"

    # Apply the train command to a list of agents: 

    log_dir = "./logs/model_progression"
    files = os.listdir(log_dir)
    files = [f for f in files if "2M" in f]
    files = sorted_alphanumeric(files)
    print("Files in logs directory:", files)
    for file in files:
        print(file)
        model_name = f"./logs/model_progression/{file}"
        print(model_name)
        rewards = train_agent_configs(configuration_file_train = configuration_file_train, configuration_file_eval = configuration_file_eval,
                                  evaluation_recording_file = recording_file, save_model = model_name, eval_freq = eval_freq, load_model = model_name,
                                  demands_list = demands_list, env_path_train = env_path_train, env_path_eval = env_path_eval,
                                  watch_train = False, watch_eval=False,  num_steps = 2e6, N = N, random_agent = False, max_evaluations= 1)