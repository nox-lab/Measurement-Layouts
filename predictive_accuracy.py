import arviz as az
import matplotlib.pyplot as plt
from typing import Callable
import numpy.typing as npt
import numpy as np
from measurement_layout_AAIO import setupModelSingle
import pandas as pd
from demands import Demands
from generating_configs_class import ConfigGenerator
import random
from particles import state_space_models as ssm
from animalai.environment import AnimalAIEnvironment
from animal_ai_reset_wrapper import AnimalAIReset
from xgboost import XGBClassifier
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION, Measurement_Layout_AAIO_precise
from collections import OrderedDict
import os 
from multipledispatch import dispatch


def logistic(x):
    return 1/(1+np.exp(-x))

def brierDecomp(preds, outs):
    brier = 1 / len(preds) * sum((preds - outs) ** 2)
    bins = np.linspace(0, 1, 11)
    binCenters = (bins[:-1] + bins[1:]) / 2
    binPredInds = np.digitize(preds, binCenters)
    binnedPreds = bins[binPredInds]

    binTrueFreqs = np.zeros(10)
    binPredFreqs = np.zeros(10)
    binCounts = np.zeros(10)

    for i in range(10):
        idx = (preds >= bins[i]) & (preds < bins[i + 1])

        binTrueFreqs[i] = np.sum(outs[idx]) / np.sum(idx) if np.sum(idx) > 0 else 0

        binPredFreqs[i] = np.mean(preds[idx]) if np.sum(idx) > 0 else 0
        binCounts[i] = np.sum(idx)

    calibration = np.sum(binCounts * (binTrueFreqs - binPredFreqs) ** 2) / np.sum(binCounts) if np.sum(
        binCounts) > 0 else 0
    refinement = np.sum(binCounts * (binTrueFreqs * (1 - binTrueFreqs))) / np.sum(binCounts) if np.sum(
        binCounts) > 0 else 0

    return brier, calibration, refinement


def extract_capabilities(layout : ssm.StateSpaceModel, folder_name: str, added_folder:str , cap_time:int = -1, full:bool = False):
    folder_name_caps = added_folder + "/" + folder_name
    if full:
        estimated_file_ext = "FULL"
    else:
        estimated_file_ext = ""
    estimated_visual = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\{folder_name_caps}\visual_est{estimated_file_ext}.npy")
    estimated_bias = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\{folder_name_caps}\bias_est{estimated_file_ext}.npy")
    try:
        noise_estimate = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\{folder_name_caps}\noise_est{estimated_file_ext}.npy") # This is the noise estimate.
    except:
        noise_estimate = np.zeros(len(estimated_visual))
        print("No noise file found, assuming no noise.")
    noise_final = noise_estimate[cap_time]



    final_capability_means = OrderedDict()
    if layout != Measurement_Layout_AAIO_NO_NAVIGATION:
        estimated_navigation = np.load(rf"C:\Users\talha\Documents\iib_projects\Measurement-Layouts\estimated_capabilities\{folder_name_caps}\navigation_est{estimated_file_ext}.npy")
        final_capability_means["navigation"] = estimated_navigation[cap_time]
    final_capability_means["visual"] = estimated_visual[cap_time]
    final_capability_means["bias"] = estimated_bias[cap_time]
    
    return final_capability_means, noise_final

def prediction_accuracy(layout : ssm.StateSpaceModel, folder_name, added_folder, model_name, N, config_modifier, seed, load_eval = True, noise_level = 0.0, cap_time = -1, N_eval=None, full = False, caps : dict = None):
    if N_eval is None:
        N_eval = N
    env_path_train = r"..\WINDOWS\AAI\Animal-AI.exe"
    env_path_eval = r"..\WINDOWS\AAI - Copy\Animal-AI.exe"
    if config_modifier == "very_precise":
        config_generator = ConfigGenerator(very_precise = True)
    elif config_modifier == "precise":
        config_generator = ConfigGenerator(precise = True)
    elif config_modifier == "closed":
        config_generator = ConfigGenerator(closed = True)
    else:
        config_generator = ConfigGenerator()
    
    T = int(len(pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward_distance"].to_numpy()) / N_eval)
    if cap_time < 0:
        cap_time = int(T) + cap_time - 1
    proportion_of_successes = np.mean(pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward"].to_numpy()[(cap_time)*N_eval:(cap_time+1)*N_eval] > -0.9)
    print(f"prop_succeses = {proportion_of_successes}")
    if not os.path.exists(rf"./csv_recordings/predictive_data/{model_name}"):
        os.makedirs(rf"./csv_recordings/predictive_data/{model_name}")
    recorded_results = rf"./csv_recordings/predictive_data/{model_name}/true_results_for_prediction.csv"   
    max_distance = np.max(pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward_distance"].to_numpy()[-N:]) # This will force in distribution.
    min_distance = np.min(pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward_distance"].to_numpy()[-N:])
    max_size = np.max(pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward_size"].to_numpy()[-N:])
    min_size = np.min(pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward_size"].to_numpy()[-N:])
    model_path = rf"./logs/{model_name}.zip"
    if not caps:  
        final_capability_means, noise_final = extract_capabilities(layout, folder_name, added_folder, cap_time = cap_time, full = full)
        cap_names = ["navigation", "visual", "bias"]
    else:
        final_capability_means = caps
        cap_names = ["ability_navigation", "ability_visual", "ability_bias_rl"]
        noise_final = 0.2
    if load_eval:
        csv_file = pd.read_csv(recorded_results)
        demands = csv_file[["Xpos", "reward_distance", "reward_size", "reward_behind"]].to_numpy()
        #demands = demands[:N] # We only want the first N because of the fact that they are repeated.
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
        yaml_string, list_of_demands = config_generator.gen_config_from_demands_batch_random(N, "example_batch_predictive.yaml", dist_max = max_distance, dist_min = min_distance, size_max = max_size, size_min = min_size, time_limit=300, numbered = False, seed = seed) # Creates yaml file with same demands as csv file.
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
    
    if not skip_model:
        from test_animal_ai import train_agent_configs
        if os.path.exists(recorded_results):
            print(f"Found results file at {recorded_results}")
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
    
    
    input_to_observations = [final_capability_means[key] for key in final_capability_means]
    print(f"INPUT TO OBSERVATIONS {input_to_observations}")
    measurement_layout_itself = layout(N, recorded_results, noiselevel = noise_level, noisy_model_performance = proportion_of_successes) 
    num_keys = len(final_capability_means)
    input_to_observations.extend([1]*num_keys)
    input_to_observations.append(noise_final)
    measurement_layout_itself.PY(x = np.array([input_to_observations]), t = None, xp = None)
    probabilities_of_success = (measurement_layout_itself.arena_outcomes).flatten()
    
    for i, probability_of_success in enumerate(probabilities_of_success):
        if i % 10 != 0:
            continue
        print(f"P(y) arena {i}: Demands are {list_of_demands[i]}.")
        
        print(f'visual_capability = {final_capability_means[cap_names[1]]}')
        if layout != Measurement_Layout_AAIO_NO_NAVIGATION:
            print(f'navigation_capability = {final_capability_means[cap_names[0]]}')
            print(f"P_nav unbiased = {logistic(final_capability_means[cap_names[0]] - distance[i]*(behind[i]*0.5+1.0))})")
        print(f'bias_capability = {final_capability_means[cap_names[2]]}')
        print(f"small_appearance = {np.log(list_of_demands[i].reward_distance/list_of_demands[i].reward_size)} , P_vis = {logistic(final_capability_means[cap_names[1]] - np.log(list_of_demands[i].reward_distance/list_of_demands[i].reward_size))}")
        print(f"Predicted probability of success {probability_of_success}, noise level {noise_final}, noisy_prob = {proportion_of_successes}")
    successes_predicted = probabilities_of_success > 0.5 
    print("proportion of predicted successes: ", np.mean(successes_predicted))

    
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
    false_positive_rate = sum((successes == 0) & (successes_predicted == 1))/len(successes)
    true_positive_rate = sum((successes == 1) & (successes_predicted == 1))/len(successes)
    true_negative_rate = sum((successes == 0) & (successes_predicted == 0))/len(successes)
    print(f"False positive rate: {false_positive_rate}")
    # false negative rate
    false_negative_rate = sum((successes == 1) & (successes_predicted == 0))/len(successes)
    print(f"False negative rate: {false_negative_rate}")
    brier_score, calibration_score, refinement_score = brierDecomp(probabilities_of_success, successes)
    print(f"brier score:{brier_score}")
    
    if cap_time < 0:
        cap_time = len(final_capability_means[cap_names[1]]) + cap_time # I hate doing this
    
    # Apply a baseline for prediction as well. Do it from the mean of the succcesses.
    #assert rewards_received[0] == csv_file["reward"].to_numpy()[-N]
    initial_evaluation_successes = pd.read_csv(rf"./csv_recordings/{folder_name}.csv")["reward"].to_numpy()[cap_time*N_eval:(cap_time+1)*N_eval] > -0.9
    successes_from_initial_eval = initial_evaluation_successes
    baseline = np.mean(successes_from_initial_eval.astype(int))
    baseline_brier_score, calibration_score, refinement_score = brierDecomp(baseline*np.ones(N), successes)
    baseline_accuracies = []
    baseline_test_accuracies = []
    baseline_FN = []
    baseline_test_FN = []
    baseline_FP = []
    baseline_test_FP = []
    baseline_TP = []
    baseline_TN = []
    for sample in range(1000):
        random_number = np.random.rand(N)
        random_number_again = np.random.rand(N)
        baseline_predicted = random_number < baseline
        baseline_test = random_number_again < baseline
        baseline_error = sum(abs(baseline_predicted.astype(int) - successes.astype(int)))
        baseline_test_error = sum(abs(baseline_test.astype(int) - baseline_predicted.astype(int)))
        baseline_accuracy = 1 - baseline_error/N
        baseline_test_accuracy = 1 - baseline_test_error/N
        baseline_accuracies.append(baseline_accuracy)
        baseline_test_accuracies.append(baseline_accuracy)
        # false positive rate
        false_positive_rate_baseline = sum((successes == 0) & (baseline_predicted == 1))/len(baseline_predicted)
        true_positive_rate_baseline = sum((successes == 1) & (baseline_predicted == 1))/len(baseline_predicted)
        baseline_TP.append(true_positive_rate_baseline)
        baseline_FP.append(false_positive_rate_baseline)
        baseline_test_FP.append(sum((baseline_predicted == 1) & (baseline_test == 0))/len(baseline_predicted))
        # false negative rate
        false_negative_rate_baseline = sum((successes == 1) & (baseline_predicted == 0))/len(baseline_predicted)
        true_negative_rate_baseline = sum((successes == 0) & (baseline_predicted == 0))/len(baseline_predicted)
        baseline_TN.append(true_negative_rate_baseline)
        baseline_FN.append(false_negative_rate_baseline)
        baseline_test_FN.append(sum((baseline_predicted == 0) & (baseline_test == 1))/len(baseline_predicted))
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
    
    
    
    # fig, ax = plt.subplots()
    # ax.hist(baseline_accuracies, alpha = 0.5, label = "Baseline accuracies", color = "b")
    # ax.hist(baseline_FP, alpha = 0.5, label = "Baseline false positive rates", color = "b")
    # ax.hist(baseline_FN, alpha = 0.5, label = "Baseline false negative rates", color = "b")
    # ax.hist(baseline_accuracies, alpha = 0.5, label = "Baseline test accuracies", color = "r")
    # ax.hist(baseline_FP, alpha = 0.5, label = "Baseline test false positive rates", color = "r")
    # ax.hist(baseline_FN, alpha = 0.5, label = "Baseline test false negative rates", color = "r")
    # ax.set_title("Baseline accuracies, false positive rates and false negative rates")
    # ax.set_xlabel("Rate")
    # ax.set_ylabel("Frequency")
    # ax.legend()
    # plt.show()
    evaluation_for_capabilities_data = pd.read_csv(rf"./csv_recordings/{folder_name}.csv")
    
    inputFeatures = ["Xpos", "reward_distance", "reward_size", "reward_behind"]
    outputFeatures = ["reward"]
    
    XTrain = evaluation_for_capabilities_data[inputFeatures].to_numpy()[-N_eval:]
    YTrain = evaluation_for_capabilities_data[outputFeatures].to_numpy()[cap_time*N_eval:(cap_time+1)*N_eval] > 0.9
    
    XTest = recorded_results[inputFeatures].to_numpy()
    YTest = recorded_results[outputFeatures].to_numpy() > 0.9
    
    model = XGBClassifier(objective='binary:logistic')
    # Set up KFold cross-validation

    model.fit(XTrain, YTrain)
    # Make predictions on the test data
    yPredictions = model.predict_proba(XTest)[:, 1]  # Get the probabilities for the positive class
    YTest = YTest.flatten()
    brierScoreXGBoost, calibrationXGBoost, refinementXGBoost = brierDecomp(yPredictions, YTest)

    print(f"XGBoost brier score: {brierScoreXGBoost}")
    XGBoost_success_prediction= yPredictions > 0.5
    XGBoost_accuracy = np.mean(XGBoost_success_prediction == YTest)
    XGBoost_FN = sum((YTest == 1) & (~XGBoost_success_prediction))/len(YTest)
    XGBoost_FP = sum((YTest == 0) & (XGBoost_success_prediction))/len(YTest)
    XGBoost_TN = sum((YTest == 0) & (~XGBoost_success_prediction))/len(YTest)
    XGBoost_TP = sum((YTest == 1) & (XGBoost_success_prediction))/len(YTest)
    print(f"XGBoost accuracy: {XGBoost_accuracy}", 
          "XGBoost FN:", XGBoost_FN, 
          "XGBoost FP: ", XGBoost_FP)
    print(f"Baseline brier score: {baseline_brier_score}")
    print(f"Model brier score: {brier_score}")
    
    original_dataframe = pd.read_csv(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv")
    
    result_data_model = {
        "model_name": model_name,
        "N_predict": N,
        "model_used" : layout.__name__,
        "initial_evaluation_success_rate": np.mean(initial_evaluation_successes),
        "predictive_evaluation_success_rate": np.mean(successes),
        "brier_score": brier_score,
        "accuracy": 1 - success_error/N,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "tpr": true_positive_rate, 
        "tnr": true_negative_rate, 
    }
    df = pd.DataFrame([result_data_model])
    # Check if the dataframe already exists in the CSV file
    df.to_csv(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv", mode='a', header=not pd.io.common.file_exists(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv"), index=False)
    result_data_XGBOOST = {
        "model_name": model_name,
        "N_predict": N,
        "model_used" : "XGBoost",
        "initial_evaluation_success_rate": np.mean(initial_evaluation_successes),
        "predictive_evaluation_success_rate": np.mean(successes),
        "brier_score": brierScoreXGBoost,
        "accuracy": XGBoost_accuracy,
        "fpr": XGBoost_FP,
        "fnr": XGBoost_FN,
        "tpr": XGBoost_TP,
        "tnr": XGBoost_TN,
    }
    
    df = pd.DataFrame([result_data_XGBOOST])
    if not original_dataframe[(original_dataframe['model_name'] == model_name) & 
                              (original_dataframe['N_predict'] == N) & 
                              (original_dataframe['model_used'] == "XGBoost")].empty:
        print("Model results already exist in the CSV file. Skipping update.")
    else:
        df.to_csv(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv", mode='a', header=not pd.io.common.file_exists(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv"), index=False)
    
    result_data_baseline = {
        "model_name": model_name,
        "N_predict": N,
        "model_used" : "proportional",
        "initial_evaluation_success_rate": np.mean(initial_evaluation_successes),
        "predictive_evaluation_success_rate": np.mean(successes),
        "brier_score": baseline_brier_score,
        "accuracy": np.mean(baseline_accuracies),
        "fpr": np.mean(baseline_FP),
        "fnr": np.mean(baseline_FN),
        "tpr": np.mean(baseline_TP),
        "tnr": np.mean(baseline_TN),
    }
    df = pd.DataFrame([result_data_baseline])
    
    if not original_dataframe[(original_dataframe['model_name'] == model_name) & 
                              (original_dataframe['N_predict'] == N) & 
                              (original_dataframe['model_used'] == "proportional")].empty:
        print("Model results already exist in the CSV file. Skipping update.")
    else:
        df.to_csv(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv", mode='a', header=not pd.io.common.file_exists(rf"./csv_recordings/predictive_data/predictive_results_for_agents.csv"), index=False)
    return brier_score, brierScoreXGBoost, baseline_brier_score