import os
import sys
import numpy as np
import pandas as pd
import json

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










