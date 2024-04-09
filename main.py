#main
# imports
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import AdamW
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from keras.optimizers import Adam
import keras
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize
import numpy as np
import optuna

# Assuming your CSVs are formatted correctly for this task
TrainingSet = np.genfromtxt("/content/drive/MyDrive/Colab Notebooks/UW/Dataset/DATASET_T2.csv", delimiter=",", skip_header=True)
ValidationSet = np.genfromtxt("/content/drive/MyDrive/Colab Notebooks/UW/Dataset/DATASET_V2.csv", delimiter=",", skip_header=True)

# Split into input (X) and outputs (Y1, Y2)
X_train = TrainingSet[:,0:5]  # Assuming the first column is the input
Y_train = [TrainingSet[:,5], TrainingSet[:,6]]  # Assuming the next two columns are outputs

X_val = ValidationSet[:,0:5]
Y_val = [ValidationSet[:,5], ValidationSet[:,6]]

# Define model architecture
input_layer = Input(shape=(5,))
hidden1 = Dense(516, activation="relu")(input_layer)
hidden2 = Dense(128, activation="relu")(hidden1)
hidden3 = Dense(64, activation="relu")(hidden2)
output1 = Dense(1, activation="linear", name='output1')(hidden3)  # First output
output2 = Dense(1, activation="linear", name='output2')(hidden3)  # Second output

model = Model(inputs=input_layer, outputs=[output1, output2])

# Compile model
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=10000,
    decay_rate=0.9)
optimizer = AdamW(learning_rate=lr_schedule)

model.compile(loss={'output1': 'mean_squared_error', 'output2': 'mean_squared_error'}, optimizer=optimizer)

# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# Fit the model
history = model.fit(X_train, {'output1': Y_train[0], 'output2': Y_train[1]},
                    validation_data=(X_val, {'output1': Y_val[0], 'output2': Y_val[1]}),
                    epochs=10000000, batch_size=100, verbose=2, callbacks=[es])

# Calculate predictions
PredTrainSet = model.predict(X_train)
PredValSet = model.predict(X_val)

# Save predictions
# Note: This will give you two arrays for each set of predictions, handle each according to your needs
np.savetxt("/content/drive/MyDrive/Colab Notebooks/UW/Dataset/T_results_output1.csv", PredTrainSet[0], delimiter=",")
np.savetxt("/content/drive/MyDrive/Colab Notebooks/UW/Dataset/T_results_output2.csv", PredTrainSet[1], delimiter=",")
np.savetxt("/content/drive/MyDrive/Colab Notebooks/UW/Dataset/V_results_output1.csv", PredValSet[0], delimiter=",")
np.savetxt("/content/drive/MyDrive/Colab Notebooks/UW/Dataset/V_results_output2.csv", PredValSet[1], delimiter=",")


# Set target, initial & bounds
target_outputs = np.array([300,600])
initial_guess = np.array([15, 0.9, 15, 2.5, 1])
bounds = [(1, 50), (0.5,0.999), (0.001, 60), (0.001, 15), (0.001, 15)]

# Assume model is your trained model
# target_outputs are the desired output values you want to achieve
# target_outputs = np.array([300,600])

# Initial guess
# initial_guess = np.array([10, 0.8, 10, 1, 1])


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
print("Best trial:")
trial = study.best_trial

print(f"Value: {trial.value}")
print("Params: ")
for key, value in trial.params.items():
    print(f"  {key}: {value}")
# Extract the best parameters
optimized_params = study.best_trial.params

# Since 'optuna' returns the parameters as a dictionary, we need to adjust them to match the 'minimize' function's expected format
optimized_options = {
    "rhobeg": optimized_params["rhobeg"],
    # "rhoend": optimized_params["rhoend"],
    "maxiter": optimized_params["maxiter"],
    "catol": optimized_params["catol"]
}

# Define bounds as constraints for COBYLA
# Assume you have 5 inputs, each with a specific range
# bounds = [(0.005, 100), (0.001,0.999), (0.001, 60), (0.001, 15), (0.001, 15)]
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
else:
    print("Optimization failed:", result.message)
     
