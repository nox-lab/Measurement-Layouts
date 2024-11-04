
# Import the necessary libraries
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from end_of_episode_callback import CustomCallback
import torch as th

import sys
import random
import pandas as pd
import os

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.environment import AnimalAIEnvironment
import subprocess

def train_agent_single_config(configuration_file_train, configuration_file_eval, env_path_train, env_path_eval, log_bool = False, aai_seed = 2023, watch_train = False, watch_eval = False, num_steps = 10000):
    
    port_train = 5005 + random.randint(
    1, 1000)
    port_eval = port_train - 1
    
    # Create the environment and wrap it...
    aai_env_train = AnimalAIEnvironment( # the environment object
        seed = aai_seed, # seed for the pseudo random generators
        file_name=env_path_train,
        arenas_configurations=configuration_file_train,
        play=False, # note that this is set to False for training
        base_port=port_train, # the port to use for communication between python and the Unity environment
        inference=watch_train, # set to True if you want to watch the agent play
        useCamera=True, # set to False if you don't want to use the camera (no visual observations)
        resolution=64,
        useRayCasts=False, # set to True if you want to use raycasts
        no_graphics= not watch_train, # set to True if you don't want to use the graphics ('headless' mode)
        timescale=5.0, # the speed at which the simulation runs
        log_folder="aailogstrain" # env logs train
    )
    print("env train created")
    print(port_eval)
    print(port_train)
    # Include callbacks
    runname = "really_short_testing_stopTRAINING" # the name of the run, used for logging
    
    aai_env_eval = AnimalAIEnvironment( # the environment object
        seed = aai_seed, # seed for the pseudo random generators
        file_name=env_path_eval,
        arenas_configurations=configuration_file_eval,
        play=False, # note that this is set to False for training
        base_port=port_eval, # the port to use for communication between python and the Unity environment
        inference = watch_eval, # set to True if you want to watch the agent play
        useCamera= True, # set to False if you don't want to use the camera (no visual observations)
        resolution=64,
        useRayCasts=False, # set to True if you want to use raycasts
        no_graphics= not watch_eval, # set to True if you don't want to use the graphics ('headless' mode)
        timescale=2.0, # the speed at which the simulation runs
        log_folder = "aailogseval" # env logs eval
    )
    print("env eval created")
    env_train = UnityToGymWrapper(aai_env_train, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True) # the wrapper for the environment
    env_eval = UnityToGymWrapper(aai_env_eval, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True) # the wrapper for the environment
    # Create aai_env_eval_new type using AnimalAIWrapper, fulfilling same interface, same as UnityToGymWrapper, 
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=0.7, verbose=1)
    eval_callback = EvalCallback(env_eval, callback_on_new_best=callback_on_best, best_model_save_path='./logs/', log_path='./logs/', eval_freq=3000, deterministic=True, render=False)
    episode_callback = CustomCallback()
    callback_list = CallbackList([eval_callback, episode_callback])

    policy_kwargs = dict(activation_fn=th.nn.ReLU) # the policy kwargs for the PPO agent, such as the activation function
    model = PPO("CnnPolicy", env_train, policy_kwargs=policy_kwargs, verbose=1) # the PPO agent
    # verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
    model.learn(num_steps, reset_num_timesteps=False, callback=callback_list) # the training loop
    env_train.close()
    env_eval.close()

# IMPORTANT! Replace the path to the application and the configuration file with the correct paths here:
env_path_train = r"..\WINDOWS\AAI\Animal-AI.exe"
env_path_eval = r"..\WINDOWS\AAI - Copy\Animal-AI.exe"
configuration_file_train = r"example.yaml"
configuration_file_eval = r"example_eval.yaml"

rewards = train_agent_single_config(configuration_file_train = configuration_file_train, configuration_file_eval = configuration_file_eval, env_path_train = env_path_train, env_path_eval = env_path_eval, watch_train = True, watch_eval=True,  num_steps = 1e4)