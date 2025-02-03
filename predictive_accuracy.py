import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import numpy as np
from measurement_layout_AAIO import setupModelSingle
import pandas as pd
from generating_configs import gen_config_from_demands_batch_random, gen_config_from_demands_batch
from demands import Demands
import random
from animalai.environment import AnimalAIEnvironment
from animal_ai_reset_wrapper import AnimalAIReset
from test_animal_ai import train_agent_configs
import os 


def logistic(x):
    return 1/(1+np.exp(-x))


if __name__ == "__main__":
    
    env_path_train = r"..\WINDOWS\AAI\Animal-AI.exe"
    env_path_eval = r"..\WINDOWS\AAI - Copy\Animal-AI.exe"
    recorded_results = r"./csv_recordings/example_batch_predictive_3_same_evals.csv"
    
    csv_name = "working_caps_predictive_3"
    estimated_visual = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\estimated_ability_visual.png_based_on_{csv_name}.npy")
    estimated_navigation = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\estimated_ability_navigation.png_based_on_{csv_name}.npy")
    estimated_bias = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\estimated_ability_bias_rl.png_based_on_{csv_name}.npy")


    final_capability_means = {
        "visual": estimated_visual[-1],
        "navigation": estimated_navigation[-1],
        "bias": estimated_bias[-1]
    }



    N = 200 
    csv_file = pd.read_csv(f"./csv_recordings/{csv_name}.csv")
    demands = csv_file[["Xpos", "reward_distance", "reward_size", "reward_behind"]].to_numpy()
    demands = demands[:N] # We only want the first N because of the fact that they are repeated.
    capability_bias = final_capability_means["bias"]
    capability_nav = final_capability_means["navigation"]
    capability_vis = final_capability_means["visual"]

    xpos = demands[:,0]
    distance = demands[:,1]
    reward_size = demands[:,2]
    behind = demands[:,3]
    # SELF DEMANDS FOR TESTING WHETHER ML PREDICTS ITSELF EVEN (I GUESS IT KIND OF DOES)
    list_of_demands = [Demands(reward_size[i], distance[i], behind[i], xpos[i]) for i in range(N)]
    #yaml_string, list_of_demands = gen_config_from_demands_batch_random(N, "example_batch_predictive.yaml", dist_min = 12, dist_max = 15, numbered = False) # Creates yaml file with same demands as csv file.
    gen_config_from_demands_batch(list_of_demands, "example_batch_predictive.yaml") # Creates yaml file with same demands as csv file. 
    # 0.695 predictive accuracy for putting the demands themselves in the yaml file, suggests that maybe something off with the measurement layout but it's okay.
    # Now let's try to use a out-of-distribution set of demands to see how well the model can predict the success of the agent out of distribution. 
    # For mix of in and out of distribution, found accuracy = 0.595. 
    # 0.69 predictive accuracy for distance between 12 and 15, nice! ML is able to somewhat predict out of distribution cases.

    rightlefteffect_ = capability_bias * xpos
    perf_nav = logistic(capability_nav - distance*(behind*0.5+1.0) + rightlefteffect_)
    perf_vis = logistic(capability_vis - np.log(distance/reward_size))
    probabilities_of_success = perf_nav*perf_vis
    successes_predicted = probabilities_of_success > 0.5 
    
    if os.path.exists(recorded_results):
        os.remove(recorded_results)
    
    train_agent_configs(configuration_file_train="example_batch_predictive.yaml", configuration_file_eval="example_batch_predictive.yaml",
                        env_path_train=env_path_train, env_path_eval=env_path_eval, evaluation_recording_file=recorded_results, demands_list = list_of_demands,
                        log_bool = False, aai_seed = 2023, watch_train = False, watch_eval = True, num_steps = 1, eval_freq = 1, save_model = False,
                        load_model = r"./logs/best_model_2.zip", N = N, max_evaluations=1)
    
    recorded_results = pd.read_csv(recorded_results)
    rewards_received = recorded_results["reward"]
    rewards_received = rewards_received.to_numpy()
    # Should be N length vector by default
    assert rewards_received.shape[0] == N
    # We want to match the elements in the vector to the success probabilities
    successes = rewards_received > -0.9
    log_likelihood = np.sum(np.log(probabilities_of_success[successes])) + np.sum(np.log(1-probabilities_of_success))
    success_error = sum(abs(successes.astype(int) - successes_predicted.astype(int)))
    print(f"Log likelihood: {log_likelihood}")
    print(f"accuracy : { 1 - success_error/N}") # 0 error corresponds to 1.0 accuracy (perfect)
    