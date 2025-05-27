
# Import the necessary libraries
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from stable_baselines3.common.vec_env.patch_gym import _patch_env
import torch as th
from evaluation_reward_ep_wrapper import EvalRewardCallback
from generating_configs import Demands
import sys
import random
import pandas as pd
import os
import numpy as np
from animal_ai_reset_wrapper import AnimalAIReset
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.vec_env import DummyVecEnv
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from animalai.environment import AnimalAIEnvironment
from generating_configs_class import ConfigGenerator
import subprocess
from extra_utils import is_image_space



def train_agent_configs(configuration_file_train, configuration_file_eval, env_path_train, env_path_eval,
                        N, evaluation_recording_file, demands_list, log_bool = False, aai_seed = 2023,
                        watch_train = False, watch_eval = False, num_steps = 10000, eval_freq = 15000, save_model = True,
                        load_model = False, max_evaluations = None, random_agent = False):
    
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
        useCamera= False, # set to False if you don't want to use the camera (no visual observations)
        resolution=64,
        useRayCasts= True, # set to True if you want to use raycasts
        raysPerSide = 2, # number of rays per side, assuming you are using raycasts
        no_graphics= False, # set to True if you don't want to use the graphics ('headless' mode)
        timescale=10, # the speed at which the simulation runs
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
        useCamera= False, # set to False if you don't want to use the camera (no visual observations)
        resolution=64,
        useRayCasts= True, # set to True if you want to use raycasts
        raysPerSide = 2, # number of rays per side, assuming you are using raycasts
        no_graphics= False, # set to True if you don't want to use the graphics ('headless' mode)
        timescale=10, # the speed at which the simulation runs
        log_folder = "aailogseval", # env logs eval
        targetFrameRate=-1
    )
    print("env eval created")
    #evaluation_set = [Demands(1, 2, 0, 0), Demands(0.1, 2, 0, 0), Demands(1, 1, 0, 1), Demands(1, 5, 0.5, 0), Demands(1, 2, 1, 0)]
    # Wrap training environment
    # --- Training Environment ---
    env_train = UnityToGymWrapper(aai_env_train, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
    # env_train = _patch_env(env_train) # Patch the environment to make it compatible with stable_baselines3
    # env_train = Monitor(env_train, filename=None, allow_early_resets=True) # Monitor the environment to log the rewards and other information
    # env_train = DummyVecEnv([lambda: env_train])
    # env_train = VecTransposeImage(env_train) # Transpose the image to match the expected input shape of the model
    # env_train = VecFrameStack(env_train, n_stack=4, channels_order='first')

    # --- Evaluation Environment ---
    env_eval = UnityToGymWrapper(aai_env_eval, uint8_visual=True, allow_multiple_obs=False, flatten_branched=True)
    # env_eval = _patch_env(env_eval) # Patch the environment to make it compatible with stable_baselines3
    # env_eval = Monitor(env_eval, filename=None, allow_early_resets=True) # Monitor the environment to log the rewards and other information
    # env_eval = DummyVecEnv([lambda: env_eval])
    # env_eval = VecTransposeImage(env_eval) # Transpose the image to match the expected input shape of the model
    # env_eval = VecFrameStack(env_eval, n_stack=4, channels_order='first')
    
    obs_train = env_train.reset()
    obs_eval = env_eval.reset()
    # print("Train obs shape:", obs_train.shape)
    # print("Eval obs shape:", obs_eval.shape)
    # Create aai_env_eval_new type using AnimalAIWrapper, fulfilling same interface, same as UnityToGymWrapper, 
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=2, verbose=1)
    print("model eval recognisedhere as well")
    eval_callback = EvalRewardCallback(evaluation_recording_file, demands_list, aai_env = aai_env_eval, eval_env = env_eval,
                                       config_file = configuration_file_eval,  callback_on_new_best=callback_on_best,
                                       best_model_path= save_model , eval_freq=eval_freq, deterministic=False, n_eval_episodes = N,
                                       num_evals = max_evaluations)
    # NEED TO CREATE AN ANNOTATED SET FOR EVALUATION USE that is used for evaluation.
    print("model callback created")
    policy_kwargs = dict(activation_fn=th.nn.ReLU) # the policy kwargs for the PPO agent, such as the activation function
    if load_model:
        # check if path exists
        if not os.path.exists(load_model):
            print("Model path does not exist")
            sys.exit("NOPE")
        model = PPO.load(load_model, env_train, tensorboard_log="./tensorboardLogs/NO_RAYCASTS_PREDDICIT")
    else:
        # For the 2M progression : model = PPO("CnnPolicy", env_train,  n_steps = 4096, batch_size = 64, clip_range=0.2, ent_coef=0.01, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboardLogsopeningymaze") # the PPO agent, HYPERPARAMETERS FROM https://arxiv.org/pdf/1909.07483
        print("model wrapping in progress")
        model = PPO("MlpPolicy", env_train,  n_steps = 4096, batch_size = 64, clip_range=0.2, ent_coef=0.01, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboardLogsFRAMESTACKED") # the PPO agent, HYPERPARAMETERS FROM https://arxiv.org/pdf/1909.07483
    
    # verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug 
    print("model eval recognised")
    model.learn(num_steps, reset_num_timesteps=False, callback=eval_callback) # the training loop
    env_train.close()
    env_eval.close()
    

# IMPORTANT! Replace the path to the application and the configuration file with the correct paths here:
# Still need to report the error
if __name__ == "__main__":
    N = 200
    N_train = 200
    config_generator = ConfigGenerator(very_precise = True)
    # Currently we have been using random demands, but we can make them SPECIAL
    demands_list_train = []
    dumb_string = config_generator.gen_config_from_demands_batch_random(N_train, filename=r"example_batch_train.yaml", time_limit=100, dist_max = 8, size_min = 0.5, size_max = 2, numbered = False, seed = 101)
    dumb_string_2, demands_list = config_generator.gen_config_from_demands_batch_random(N, r"example_batch_eval.yaml", time_limit=150, dist_max=15, size_min = 0.5, size_max = 2, numbered = False, seed = 1)
    eval_freq = 12500
    env_path_train = r"..\WINDOWS\AAI\Animal-AI.exe"
    env_path_eval = r"..\WINDOWS\AAI - Copy\Animal-AI.exe"
    configuration_file_train = r"example_batch_train.yaml"  # !!!!! ODD NUMBER OF ARENAS REQUIRED skips arenas for some reason !!!!!
    configuration_file_eval = r"example_batch_eval.yaml"
    model_name = r"logs/raycasts_alone_500k/raycasts_alone_500k"
    recording_file = r"csv_recordings/raycasts_alone_500k.csv"
    rewards = train_agent_configs(configuration_file_train = configuration_file_train, configuration_file_eval = configuration_file_eval,
                                  evaluation_recording_file = recording_file, save_model = model_name, eval_freq = eval_freq,
                                  demands_list = demands_list, env_path_train = env_path_train, env_path_eval = env_path_eval,
                                  watch_train = False, watch_eval=True,  num_steps = 5e5, N = N, random_agent = False)