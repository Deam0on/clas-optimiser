# main
# imports
import os
import sys
import numpy as np
import optuna
import pandas as pd
import matplotlib.pyplot as plt

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

# Load model
model = load_model('/home/deamoon_uw_nn/bucket_source/my_model.h5')  # Loads the model

# Set target, initial & bounds
target_outputs = np.array([300,600])
initial_guess = np.array([15, 0.9, 15, 2.5, 1])
bounds = [(1, 50), (0.5,0.999), (0.001, 60), (0.001, 15), (0.001, 15)]

def objective_function(inputs):
    # Reshape inputs to match the model's expected input shape
    inputs_reshaped = inputs.reshape(1, -1)
    # Predict outputs based on the current inputs
    predicted_outputs = model.predict(inputs_reshaped)
    # Calculate the difference (error) between predicted and target outputs
    error = np.sum((predicted_outputs - target_outputs)**2)
    return error

def optimize_with_cobyla(trial):
    rhobeg = trial.suggest_float("rhobeg", 0.1, 2.0)
    # rhoend = trial.suggest_float("rhoend", 1e-6, 1e-2)
    maxiter = trial.suggest_int("maxiter", 100, 10000)
    catol = trial.suggest_float("catol", 1e-4, 1e-1)

    # Define bounds as constraints for COBYLA
    # bounds = [(0.005, 100), (0.001,0.999), (0.001, 60), (0.001, 15), (0.001, 15)]
    constraints = [{'type': 'ineq', 'fun': lambda x, lb=lb, ub=ub: ub - x[i]} for i, (lb, ub) in enumerate(bounds)]
    constraints += [{'type': 'ineq', 'fun': lambda x, lb=lb, ub=ub: x[i] - lb} for i, (lb, ub) in enumerate(bounds)]

    result = minimize(objective_function, initial_guess, method='COBYLA',
                      options={'rhobeg': rhobeg, 'maxiter': maxiter, 'catol': catol},
                      constraints=constraints)

    # Return the final value of the objective function as the metric to minimize
    return result.fun

# Define the optimization study
study = optuna.create_study(direction='minimize')
study.optimize(optimize_with_cobyla, n_trials=10)

# Print the optimization results
trial = study.best_trial
optimized_params = study.best_trial.params

# Since 'optuna' returns the parameters as a dictionary, we need to adjust them to match the 'minimize' function's expected format
optimized_options = {
    "rhobeg": optimized_params["rhobeg"],
    "maxiter": optimized_params["maxiter"],
    "catol": optimized_params["catol"]
}

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
