import os
import sys
import numpy as np
import pandas as pd

def extract_variables_from_csv(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, header=None, names=['Variable', 'Value'])
    
    # Extract the target outputs
    target_outputs = np.array([data.iloc[0, 1], data.iloc[1, 1]])

    # Extract the initial guess values (assuming they are in the first set of variable listings)
    initial_guess_indices = [2, 3, 4, 5, 6]  # indices of the initial guesses based on your CSV snippet
    initial_guess = np.array([data.iloc[i, 1] for i in initial_guess_indices])

    # Extract bounds (assuming each variable's bounds are in sequence following the initial guesses)
    bounds = []
    for i in range(len(initial_guess_indices)):
        min_index = 7 + 2*i  # starting index for min bounds based on your CSV snippet
        max_index = min_index + 1
        bounds.append((data.iloc[min_index, 1], data.iloc[max_index, 1]))
    
    return target_outputs, initial_guess, np.array(bounds, dtype=float)

os.system("gsutil -m cp gs://uw-nn-storage_v2/ASP/Upload/nn_push.csv /home/deamoon_uw_nn/bucket_source")
file_path = '/home/deamoon_uw_nn/bucket_source/nn_push.csv'
target_outputs, initial_guess, bounds = extract_variables_from_csv(file_path)
print("Target Outputs:", target_outputs)
print("Initial Guess:", initial_guess)
print("Bounds:", bounds)
