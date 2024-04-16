# main
# imports
import os
import sys
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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

# Load model
# model = load_model('/home/deamoon_uw_nn/bucket_source/uw_nn.h5')  # Loads the model
model = tensorflow.keras.models.load_model('/home/deamoon_uw_nn/bucket_source/uw_nn.keras')  # Loads the model

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

# Usage example
os.system("gsutil -m cp gs://uw-nn-storage_v2/ASP/Upload/nn_push.csv /home/deamoon_uw_nn/bucket_source")
csv_to_json('/home/deamoon_uw_nn/bucket_source/nn_push.csv', '/home/deamoon_uw_nn/bucket_source/nn_push.json')
target_outputs, initial_guess = load_data_and_create_arrays('/home/deamoon_uw_nn/bucket_source/nn_push.json')


# target_outputs = np.array([300,600])
# initial_guess = np.array([15, 0.9, 15, 2.5, 1])

bounds = [(1, 50), (0.5,0.999), (0.001, 60), (0.001, 20), (0.001, 20)]

def objective_function(inputs):
    # Reshape inputs to match the model's expected input shape
    inputs_reshaped = inputs.reshape(1, -1)
    # Predict outputs based on the current inputs
    predicted_outputs = model.predict(inputs_reshaped)
    # Calculate the difference (error) between predicted and target outputs
    error = np.sum((predicted_outputs - target_outputs)**2)
    return error

# Load the optimized parameters from a file
with open('/home/deamoon_uw_nn/bucket_source/optimized_params.json', 'r') as f:
    optimized_params = json.load(f)

# Define bounds as constraints for COBYLA
# Convert bounds to constraints for COBYLA
def constraint_func(params, index, bound, lower=True):
    """Generates a function to enforce lower or upper bounds."""
    if lower:
        return params[index] - bound  # For lower bound
    else:
        return bound - params[index]  # For upper bound

constraints = []
for i, (lower_bound, upper_bound) in enumerate(bounds):
    # Lower bound constraint for each parameter
    constraints.append({'type': 'ineq', 'fun': constraint_func, 'args': (i, lower_bound, True)})
    # Upper bound constraint for each parameter
    constraints.append({'type': 'ineq', 'fun': constraint_func, 'args': (i, upper_bound, False)})


# Run optimization with COBYLA using optimized parameters to find inputs that match the target outputs
result = minimize(objective_function, initial_guess, method='COBYLA',
                  options=optimized_params, constraints=constraints)

if result.success:
    optimal_inputs = np.round(result.x, 2)
    print("Optimal inputs that lead to desired outputs:", optimal_inputs)
    np.savetxt("/home/deamoon_uw_nn/bucket_source/opti_res.csv", optimal_inputs, delimiter=",")
else:
    print("Optimization failed:", result.message)
