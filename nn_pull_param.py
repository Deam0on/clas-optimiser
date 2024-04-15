import os
import sys
import numpy as np
import pandas as pd

def extract_variables_from_csv(file_path):
    # Load the CSV file
    data = pd.read_csv(file_path, header=None, names=['Variable', 'Value'])
    # Define the keys for each category
    target_outputs_keys = ['M_V', 'M_N']
    initial_guess_keys = ['F_total', 'AS/T', 'C(API)', 'C(SDS)', 'F_min']
    bounds_keys = {'F_total','AS/T','C(API)','C(SDS)','F_min'}
    
    # Extract the target outputs
    target_outputs = np.array([data[data['Variable'] == key]['Value'].astype(float).values[0] for key in target_outputs_keys])
    # Extract the initial guess values
    initial_guess = np.array([data[data['Variable'] == key]['Value'].astype(float).values[0] for key in initial_guess_keys])
    # Extract bounds using the bounds_keys dictionary
    bounds = [bounds_keys[key] for key in initial_guess_keys]
    
    return target_outputs, initial_guess, bounds

os.system("gsutil -m cp gs://uw-nn-storage_v2/ASP/Upload/nn_push.csv /home/deamoon_uw_nn/bucket_source")
file_path = '/home/deamoon_uw_nn/bucket_source/nn_push.csv'
target_outputs, initial_guess, bounds = extract_variables_from_csv(file_path)
print("Target Outputs:", target_outputs)
print("Initial Guess:", initial_guess)
print("Bounds:", bounds)
