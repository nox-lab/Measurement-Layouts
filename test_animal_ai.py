
# Import the necessary libraries
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
import subprocess



def train_agent_configs(configuration_file_train, configuration_file_eval, env_path_train, env_path_eval, N, evaluation_recording_file, demands_list, log_bool = False, aai_seed = 2023, watch_train = False, watch_eval = False, num_steps = 10000, eval_freq = 15000, save_model = True, load_model = False, max_evaluations = None):
    
    port_train = 5005 + random.randint(
    1, 1000)
    port_eval = port_train - 1

    np.random.seed(0)
    # Create the environment and wrap it...
    
    aai_env_train = AnimalAIEnvironment( # the environment object
        seed = aai_seed, # seed for the pseudo random generators
        file_name=env_path_train,
        arenas_configurations=configuration_file_train,
        play=False, # note that this is set to False for training
        base_port=port_train, # the port to use for communication between python and the Unity environment
        inference=watch_train, # set to True if you want to watch the agent play
        useCamera= True, # set to False if you don't want to use the camera (no visual observations)
        resolution=64,
        useRayCasts=True, # set to True if you want to use raycasts
        no_graphics= False, # set to True if you don't want to use the graphics ('headless' mode)
        timescale=5.0, # the speed at which the simulation runs
        log_folder="aailogstrain", # env logs train
        targetFrameRate= -1 # no limit on frame rate, fast as possible.
    )
    print("env train created")
    print(port_eval)
    print(port_train)
    # Include callbacks
    runname = "really_short_testing_stopTRAINING" # the name of the run, used for logging 
    
    aai_env_eval = AnimalAIReset( # the environment object
        seed = aai_seed, # seed for the pseudo random generators
        file_name=env_path_eval,
        arenas_configurations=configuration_file_eval,
        play=False, # note that this is set to False for training
        base_port=port_eval, # the port to use for communication between python and the Unity environment
        inference = watch_eval, # set to True if you want to watch the agent play
        useCamera= True, # set to False if you don't want to use the camera (no visual observations)
        resolution=64,
        useRayCasts=True, # set to True if you want to use raycasts
        no_graphics= False, # set to True if you don't want to use the graphics ('headless' mode)
        timescale=5.0, # the speed at which the simulation runs
        log_folder = "aailogseval", # env logs eval
        targetFrameRate=-1
    )
    print("env eval created")
    #evaluation_set = [Demands(1, 2, 0, 0), Demands(0.1, 2, 0, 0), Demands(1, 1, 0, 1), Demands(1, 5, 0.5, 0), Demands(1, 2, 1, 0)]
    env_train = UnityToGymWrapper(aai_env_train, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True) # the wrapper for the environment
    env_eval = UnityToGymWrapper(aai_env_eval, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True) # the wrapper for the environment
    # Create aai_env_eval_new type using AnimalAIWrapper, fulfilling same interface, same as UnityToGymWrapper, 
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2, verbose=1)
    eval_callback = EvalRewardCallback(evaluation_recording_file, demands_list, aai_env = aai_env_eval, eval_env = env_eval,
                                       config_file = configuration_file_eval,  callback_on_new_best=callback_on_best,
                                       best_model_path= save_model , eval_freq=eval_freq, deterministic=False, n_eval_episodes = N,
                                       num_evals = max_evaluations)
    # NEED TO CREATE AN ANNOTATED SET FOR EVALUATION USE that is used for evaluation.
   
    policy_kwargs = dict(activation_fn=th.nn.ReLU) # the policy kwargs for the PPO agent, such as the activation function
    if load_model:
        model = PPO.load(load_model, env_train, tensorboard_log="./tensorboardLogsopeningymaze")
    else:
        model = PPO("CnnPolicy", env_train, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboardLogsopeningymaze") # the PPO agent, HYPERPARAMETERS FROM https://arxiv.org/pdf/1909.07483
    # verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
    model.learn(num_steps, reset_num_timesteps=False, callback=eval_callback) # the training loop
    env_train.close()
    env_eval.close()

# IMPORTANT! Replace the path to the application and the configuration file with the correct paths here:
# Still need to report the error
if __name__ == "__main__":
    N = 200
    N_train = 500
    training_set, training_demands = gen_config_from_demands_batch_random(N_train, r"example_batch_train.yaml", time_limit=75, size_min = 0.2, dist_max = 10, numbered = False) # the training set, a list of Demands objects, writes to a file.
    evaluation_set, demands_list = gen_config_from_demands_batch_random(N, r"example_batch_eval.yaml", time_limit=75, size_min = 0.2, dist_max = 15, numbered = False) # the evaluation set, a list of Demands objects, writes to a file.
    env_path_train = r"..\WINDOWS\AAI\Animal-AI.exe"
    env_path_eval = r"..\WINDOWS\AAI - Copy\Animal-AI.exe"
    configuration_file_train = r"example_batch_train.yaml"  # !!!!! ODD NUMBER OF ARENAS REQUIRED skips arenas for some reason !!!!!
    configuration_file_eval = r"example_batch_eval.yaml"
    model_name = r"./logs/best_model_4.zip"
    recording_file = r"./csv_recordings/example_batch_predictive_4_harder_train10eval15.csv"
    rewards = train_agent_configs(configuration_file_train = configuration_file_train, configuration_file_eval = configuration_file_eval,
                                  evaluation_recording_file = recording_file, save_model = model_name,
                                  demands_list = demands_list, env_path_train = env_path_train, env_path_eval = env_path_eval,
                                  watch_train = True, watch_eval=True,  num_steps = 1e6, N = N)