
# Import the necessary libraries
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor

import torch as th

import sys
import random
import pandas as pd
import os

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.environment import AnimalAIEnvironment
import subprocess

def train_agent_single_config(configuration_file, env_path , log_bool = False, aai_seed = 2023, watch = False, num_steps = 10000, num_eval = 100):
    
    port = 5005 + random.randint(
    0, 1000
    )  # uses a random port to avoid problems if a previous version exits slowly
    
    # Create the environment and wrap it...
    aai_env = AnimalAIEnvironment( # the environment object
        seed = aai_seed, # seed for the pseudo random generators
        file_name=env_path,
        arenas_configurations=configuration_file,
        play=False, # note that this is set to False for training
        base_port=port, # the port to use for communication between python and the Unity environment
        inference=watch, # set to True if you want to watch the agent play
        useCamera=True, # set to False if you don't want to use the camera (no visual observations)
        resolution=64,
        useRayCasts=False, # set to True if you want to use raycasts
        no_graphics=False, # set to True if you don't want to use the graphics ('headless' mode)
        timescale=5
    )

    env = UnityToGymWrapper(aai_env, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True) # the wrapper for the environment
    
    runname = "closed_45_angle_straight_path" # the name of the run, used for logging

    policy_kwargs = dict(activation_fn=th.nn.ReLU) # the policy kwargs for the PPO agent, such as the activation function
    
    model = PPO("CnnPolicy", env,  n_steps = 4096, batch_size = 64, clip_range=0.2, ent_coef=0.01, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboardLogs" + runname) # the PPO agent, HYPERPARAMETERS FROM https://arxiv.org/pdf/1909.07483    # verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
    
    model.learn(num_steps, reset_num_timesteps=False)
    
    env.close()

# IMPORTANT! Replace the path to the application and the configuration file with the correct paths here:
env_path = r"..\WINDOWS\AAI\Animal-AI.exe"
configuration_file = r"testing_imermediate_OPEN.yaml"

rewards = train_agent_single_config(configuration_file=configuration_file, env_path = env_path, watch = True, num_steps = 50000, num_eval = 3000)