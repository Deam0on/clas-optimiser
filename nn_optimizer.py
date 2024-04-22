# main
# imports
import os
import sys
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import schedules
# from tensorflow.keras.optimizers import AdamW
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Dense, Input
# from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
from keras.models import load_model
import json

from multiprocessing import Pool

# import time
# from tqdm import tqdm

# def trial_time_callback(study, trial):
#     if not hasattr(study, 'trial_times'):
#         study.trial_times = {}
#     if trial.number not in study.trial_times:
#         study.trial_times[trial.number] = {}
#         study.trial_times[trial.number]['start'] = time.time()
#     else:
#         study.trial_times[trial.number]['end'] = time.time()
#         duration = study.trial_times[trial.number]['end'] - study.trial_times[trial.number]['start']
#         study.trial_times[trial.number]['duration'] = duration
#         print(f"Trial {trial.number} completed in {duration:.2f} seconds.")

# def optuna_optimize_function(n_trials=10):
#     for _ in tqdm(range(n_trials), desc="Optimizing"):
#         study.optimize(optimize_with_cobyla, n_trials=1, callbacks=[trial_time_callback])

def csv_to_json(csv_file_path, json_file_path):
    # Load the CSV data
    data = pd.read_csv(csv_file_path, header=None, names=['Key', 'Value'])
    
    # Convert to a dictionary
    data_dict = dict(zip(data['Key'], data['Value']))
    
    # Save as JSON
    with open(json_file_path, 'w') as json_file:
        json.dump(data_dict, json_file)

def load_data_and_create_arrays(json_file_path):
    # Load JSON data
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract arrays based on keys
    target_outputs = np.array([data['M_V'], data['M_N']])
    initial_guess = np.array([data['F_total'], data['AS/T'], data['C(API)'], data['C(SDS)'], data['C(HPMC)']])  # replace 'another_key' as needed

    return target_outputs, initial_guess

def objective_function(inputs):
    # Reshape inputs to match the model's expected input shape
    inputs_reshaped = inputs.reshape(1, -1)
    # Predict outputs based on the current inputs
    predicted_outputs = model.predict(inputs_reshaped)
    # Calculate the difference (error) between predicted and target outputs
    error = np.sum((predicted_outputs - target_outputs)**2)
    return error

def optimize_with_cobyla(trial):
    rhobeg = trial.suggest_float("rhobeg", 0.1, 1.0)
    # rhoend = trial.suggest_float("rhoend", 1e-6, 1e-2)
    maxiter = trial.suggest_int("maxiter", 100, 10000)
    catol = trial.suggest_float("catol", 1e-4, 1e-2)

    # Define bounds as constraints for COBYLA
    # bounds = [(0.005, 100), (0.001,0.999), (0.001, 60), (0.001, 15), (0.001, 15)]
    constraints = [{'type': 'ineq', 'fun': lambda x, lb=lb, ub=ub: ub - x[i]} for i, (lb, ub) in enumerate(bounds)]
    constraints += [{'type': 'ineq', 'fun': lambda x, lb=lb, ub=ub: x[i] - lb} for i, (lb, ub) in enumerate(bounds)]

    result = minimize(objective_function, initial_guess, method='COBYLA',
                      options={'rhobeg': rhobeg, 'maxiter': maxiter, 'catol': catol},
                      constraints=constraints)

    # Return the final value of the objective function as the metric to minimize
    return result.fun


if __name__ == '__main__':

    # Load model
    # model = load_model('/home/deamoon_uw_nn/bucket_source/uw_nn.h5')  # Loads the model
    model = load_model('/home/deamoon_uw_nn/bucket_source/uw_nn.keras')  # Loads the model

    # Usage example
    os.system("gsutil -m cp gs://uw-nn-storage_v2/ASP/Upload/nn_push.csv /home/deamoon_uw_nn/bucket_source")
    csv_to_json('/home/deamoon_uw_nn/bucket_source/nn_push.csv', '/home/deamoon_uw_nn/bucket_source/nn_push.json')
    target_outputs, initial_guess = load_data_and_create_arrays('/home/deamoon_uw_nn/bucket_source/nn_push.json')
    
    # target_outputs = np.array([300,600])
    # initial_guess = np.array([15, 0.9, 15, 2.5, 1])

    bounds = [(1, 50), (0.5,0.999), (0.001, 60), (0.001, 20), (0.001, 20)]
    
    # Define the optimization study
    # study = optuna.create_study(direction='minimize')
    study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler(), direction='minimize')
    # study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')

    # with Pool() as p:
    #     p.map(optuna_optimize_function, [10]*4)  # Assuming you want to run 10 trials on 4 different processes
    
    with Pool() as p:
        # p.map(study.optimize(optimize_with_cobyla, n_trials=10, n_jobs=4, callbacks=[trial_time_callback]))
        p.map(study.optimize(optimize_with_cobyla, n_trials=10, n_jobs=-1))
    
    # Print the optimization results
    trial = study.best_trial
    optimized_params = study.best_trial.params
    
    # Since 'optuna' returns the parameters as a dictionary, we need to adjust them to match the 'minimize' function's expected format
    optimized_options = {
        "rhobeg": optimized_params["rhobeg"],
        "maxiter": optimized_params["maxiter"],
        "catol": optimized_params["catol"]
    }
    
    # Dump the optimized parameters into a json file
    with open('/home/deamoon_uw_nn/bucket_source/optimized_params.json', 'w') as json_file:
        json.dump(optimized_options, json_file)
