import numpy as np
import pandas as pd
from demands import Demands
from generating_configs_class import ConfigGenerator
from particles import state_space_models as ssm
from xgboost import XGBClassifier
from various_measurement_layouts import Measurement_Layout_AAIO, Measurement_Layout_AAIO_NO_NAVIGATION, Measurement_Layout_AAIO_precise
from collections import OrderedDict
import os 
import json
from extra_utils import sorted_alphanumeric

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


def prediction_accuracy(evaluation_for_capabilities_data, recorded_results):
    

    
    inputFeatures = ["Xpos", "reward_distance", "reward_size", "reward_behind"]
    outputFeatures = ["reward"]
    
    XTrain = evaluation_for_capabilities_data[inputFeatures].to_numpy()
    
    YTrain = evaluation_for_capabilities_data[outputFeatures].to_numpy() > -0.9
    
    XTest = recorded_results[inputFeatures].to_numpy()
    YTest = recorded_results[outputFeatures].to_numpy() > -0.9
    
    model = XGBClassifier(objective='binary:logistic')
    # Set up KFold cross-validation

    model.fit(XTrain, YTrain)
    # Make predictions on the test data
    yPredictions = model.predict_proba(XTest)[:, 1]  # Get the probabilities for the positive class
    print(yPredictions)
    YTest = YTest.flatten()
    print(YTest)
    brierScoreXGBoost, calibrationXGBoost, refinementXGBoost = brierDecomp(yPredictions, YTest)

    print(f"XGBoost brier score: {brierScoreXGBoost}")
    XGBoost_success_prediction= yPredictions > 0.5
    print(f"XGBoost success prediction: {XGBoost_success_prediction}")
    XGBoost_accuracy = np.mean(XGBoost_success_prediction == YTest)
    XGBoost_FN = sum((YTest == 1) & (~XGBoost_success_prediction))/len(YTest)
    XGBoost_FP = sum((YTest == 0) & (XGBoost_success_prediction))/len(YTest)
    XGBoost_TN = sum((YTest == 0) & (~XGBoost_success_prediction))/len(YTest)
    XGBoost_TP = sum((YTest == 1) & (XGBoost_success_prediction))/len(YTest)
    print(f"XGBoost accuracy: {XGBoost_accuracy}", 
          "XGBoost FN:", XGBoost_FN, 
          "XGBoost FP: ", XGBoost_FP)
    
    return brierScoreXGBoost
    
folder_name_this = r"camera_with_frame_stacking_400k"
models_folder_path = fr"./logs/{folder_name_this}/"
files_list = sorted_alphanumeric(os.listdir(models_folder_path))
true_XGBOOST_brier_scores = []
for time_point in range(len(files_list)):
    print(f"Evaluating time point {time_point} with file {files_list[time_point]}")
    specific_file = files_list[time_point][:-4]
    N_eval = 200



    recorded_results_predictive = pd.read_csv(rf"./csv_recordings/predictive_data/{folder_name_this}/{specific_file}/true_results_for_prediction.csv")
    evaluation_for_capabilities_data = pd.read_csv(rf"./csv_recordings/{folder_name_this}.csv")[(time_point)*N_eval: (time_point+1)*N_eval]
    print(evaluation_for_capabilities_data.shape)

    final_BS = prediction_accuracy(evaluation_for_capabilities_data=evaluation_for_capabilities_data, 
                    recorded_results=recorded_results_predictive)
    true_XGBOOST_brier_scores.append(final_BS)
print("True XGBoost Brier Scores:", true_XGBOOST_brier_scores)
with open(rf"true_XGBOOST_brier_scores_{folder_name_this}.json", "w") as f:
    json.dump(true_XGBOOST_brier_scores, f)
