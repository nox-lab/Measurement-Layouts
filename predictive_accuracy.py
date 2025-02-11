import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import numpy as np
from measurement_layout_AAIO import setupModelSingle
import pandas as pd
from generating_configs import gen_config_from_demands_batch_random, gen_config_from_demands_batch
from demands import Demands
from generating_configs_class import ConfigGenerator
import random
from animalai.environment import AnimalAIEnvironment
from animal_ai_reset_wrapper import AnimalAIReset

import os 


def logistic(x):
    return 1/(1+np.exp(-x))


if __name__ == "__main__":
    env_path_train = r"..\WINDOWS\AAI\Animal-AI.exe"
    env_path_eval = r"..\WINDOWS\AAI - Copy\Animal-AI.exe"
    
    config_generator = ConfigGenerator(precise = False)
    N = 200
    model_name = "best_model_5_precise"
    folder_name = "working_caps_predictive_5_harder_train8eval10_precise"
    load_eval = False
    if not os.path.exists(rf"./csv_recordings/predictive_data/{model_name}"):
        os.makedirs(rf"./csv_recordings/predictive_data/{model_name}")
    recorded_results = rf"./csv_recordings/predictive_data/{model_name}/true_results_for_prediction.csv"   
    max_distance = np.max(pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward_distance"].to_numpy()[-N:]) # This will force in distribution.
    estimated_visual = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\{folder_name}\visual_est.npy")
    estimated_navigation = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\{folder_name}\navigation_est.npy")
    estimated_bias = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\{folder_name}\bias_est.npy")

    # Seems to do better with a non-deterministic agent.
    model_path = rf"./logs/{model_name}.zip"



    final_capability_means = {
        "visual": estimated_visual[-1],
        "navigation": estimated_navigation[-1],
        "bias": estimated_bias[-1]
    }



    capability_bias = final_capability_means["bias"]
    capability_nav = final_capability_means["navigation"]
    capability_vis = final_capability_means["visual"]
    if load_eval:
        csv_file = pd.read_csv(recorded_results)
        demands = csv_file[["Xpos", "reward_distance", "reward_size", "reward_behind"]].to_numpy()
        demands = demands[:N] # We only want the first N because of the fact that they are repeated.
        print("Using evaluations already made")
        skip_model = True
        xpos = demands[:,0]
        distance = demands[:,1]
        reward_size = demands[:,2]
        behind = demands[:,3]
        # SELF DEMANDS FOR TESTING WHETHER ML PREDICTS ITSELF EVEN (I GUESS IT KIND OF DOES)
        list_of_demands = [Demands(reward_size[i], distance[i], behind[i], xpos[i]) for i in range(N)]
    else:
        skip_model = False
        yaml_string, list_of_demands = config_generator.gen_config_from_demands_batch_random(N, "example_batch_predictive.yaml", dist_max = max_distance, numbered = False) # Creates yaml file with same demands as csv file.
        xpos = np.array([demand.Xpos for demand in list_of_demands])
        distance = np.array([demand.reward_distance for demand in list_of_demands])
        reward_size = np.array([demand.reward_size for demand in list_of_demands])
        behind = np.array([demand.reward_behind for demand in list_of_demands])
    # gen_config_from_demands_batch(list_of_demands, "example_batch_predictive.yaml") # Creates yaml file with same demands as csv file. 
    # 0.695 predictive accuracy for putting the demands themselves in the yaml file, suggests that maybe something off with the measurement layout but it's okay.
    # Now let's try to use a out-of-distribution set of demands to see how well the model can predict the success of the agent out of distribution. 
    # For mix of in and out of distribution, found accuracy = 0.595. 
    # 0.69 predictive accuracy for distance between 12 and 15, nice! ML is able to somewhat predict out of distribution cases.
    # 0.57 prediciive accuracy for new OOD, this one actually took consideration of the behind rewards as well, unlikw previous. So I guess still something. 
    # 0.8 accuracy with random new tested agent, working backwards movement this time. 
    rightlefteffect_ = capability_bias * xpos
    perf_nav = logistic(capability_nav - distance*(behind*0.5+1.0) + rightlefteffect_)
    perf_vis = logistic(capability_vis - np.log(distance/reward_size))
    probabilities_of_success = perf_nav*perf_vis
    for i, probability_of_success in enumerate(probabilities_of_success):
        print(f"Probability of success for arena {i}: {probability_of_success}. Demands are {list_of_demands[i]}")
        
    successes_predicted = probabilities_of_success > 0.5 
    print("proportion of predicted successes: ", np.mean(successes_predicted))
    

            
    if not skip_model:
        from test_animal_ai import train_agent_configs
        if os.path.exists(recorded_results):
            answer = input("Do you want to delete the recorded results file? (y/n)")
            if answer == "y":
                os.remove(recorded_results)
            else:
                print("Abortin")
                exit()
        print(f"Using model at {model_path}")
        train_agent_configs(configuration_file_train="example_batch_predictive.yaml", configuration_file_eval="example_batch_predictive.yaml",
                            env_path_train=env_path_train, env_path_eval=env_path_eval, evaluation_recording_file=recorded_results, demands_list = list_of_demands,
                            log_bool = False, aai_seed = 2023, watch_train = False, watch_eval = True, num_steps = 1, eval_freq = 1, save_model = False,
                            load_model = model_path, N = N, max_evaluations=1)
    
    recorded_results = pd.read_csv(recorded_results)
    rewards_received = recorded_results["reward"]
    rewards_received = rewards_received.to_numpy()
    
    
    # Should be N length vector by default
    assert rewards_received.shape[0] == N
    # We want to match the elements in the vector to the success probabilities
    successes = rewards_received > -0.9
    print("Proportion of successes: ", np.mean(successes))
    log_likelihood = np.sum(np.log(probabilities_of_success[successes])) + np.sum(np.log(1-probabilities_of_success))
    print("Comparing the predicted successes to the actual successes")
    print("Mean of this comparison: ", np.mean(successes == successes_predicted))
    success_error = sum(abs(successes.astype(int) - successes_predicted.astype(int)))
    print(f"Log likelihood: {log_likelihood}")
    print(f"accuracy : { 1 - success_error/N}") # 0 error corresponds to 1.0 accuracy (perfect)
    # false positive rate
    false_positive_rate = sum((successes == 0) & (successes_predicted == 1))/sum(successes == 0)
    print(f"False positive rate: {false_positive_rate}")
    # false negative rate
    false_negative_rate = sum((successes == 1) & (successes_predicted == 0))/sum(successes == 1)
    print(f"False negative rate: {false_negative_rate}")
    brier_score = np.mean((successes.astype(int) - probabilities_of_success)**2)
    print(f"brier score:{brier_score}")
    
    
    # Apply a baseline for prediction as well. Do it from the mean of the succcesses.
    #assert rewards_received[0] == csv_file["reward"].to_numpy()[-N]
    successes_from_initial_eval = rewards_received > -0.9
    baseline = np.mean(successes_from_initial_eval.astype(int))
    baseline_brier_score = np.mean((successes_from_initial_eval.astype(int) - baseline)**2)
    baseline_accuracies = []
    baseline_FN = []
    baseline_FP = []
    for sample in range(1000):
        random_number = np.random.rand(N)
        baseline_predicted = random_number < baseline
        baseline_error = sum(abs(baseline_predicted.astype(int) - successes.astype(int)))
        baseline_accuracy = 1 - baseline_error/N
        baseline_accuracies.append(baseline_accuracy)
        # false positive rate
        false_positive_rate_baseline = sum((successes == 0) & (baseline_predicted == 1))/sum(successes == 0)
        baseline_FP.append(false_positive_rate_baseline)
        # false negative rate
        false_negative_rate_baseline = sum((successes == 1) & (baseline_predicted == 0))/sum(successes == 1)
        baseline_FN.append(false_negative_rate_baseline)
    initial_evaluation_successes = pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward"].to_numpy()[-N:] > -0.9
    print(initial_evaluation_successes.shape)
    print("Success rate during initial evaluation: ", np.mean(initial_evaluation_successes))
    print("Success rate in predictive arenas", np.mean(successes))
    print("Baseline brier score: ", baseline_brier_score)
    print("Model brier score: ", brier_score)
    print("Baseline accuracy: ", np.mean(baseline_accuracies))
    print("Variance of baseline accuracies: ", np.var(baseline_accuracies))
    
    if baseline_brier_score > brier_score:
        print("Model is better than baseline")
        print("good")
    
    result_data = {
        "initial_evaluation_success_rate": np.mean(initial_evaluation_successes),
        "accuracy": 1 - success_error/N,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "brier_score": brier_score,
        "baseline_brier_score": baseline_brier_score,
        "baseline_accuracy": np.mean(baseline_accuracies),
        "baseline_accuracy_variance": np.var(baseline_accuracies),
        "baseline_FP": np.mean(baseline_FP),
        "baseline_FN": np.mean(baseline_FN),
    }
    df = pd.DataFrame([result_data])
    df.to_csv(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv", mode='a', header=not pd.io.common.file_exists(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv"), index=False)
    
    fig, ax = plt.subplots()
    ax.hist(baseline_accuracies, alpha = 0.5, label = "Baseline accuracies")
    ax.hist(baseline_FP, alpha = 0.5, label = "Baseline false positive rates")
    ax.hist(baseline_FN, alpha = 0.5, label = "Baseline false negative rates")
    ax.set_title("Baseline accuracies, false positive rates and false negative rates")
    ax.set_xlabel("Rate")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()