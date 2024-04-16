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

# os.system("gcloud storage cp gs://uw-nn-storage/DATASET_T2.csv /home/deamoon_uw_nn/bucket_source")
# os.system("gcloud storage cp gs://uw-nn-storage/DATASET_V2.csv /home/deamoon_uw_nn/bucket_source")

# Assuming your CSVs are formatted correctly for this task
TrainingSet = np.genfromtxt("/home/deamoon_uw_nn/bucket_source/DATASET_T2.csv", delimiter=",", skip_header=True)
ValidationSet = np.genfromtxt("/home/deamoon_uw_nn/bucket_source/DATASET_V2.csv", delimiter=",", skip_header=True)

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
# optimizer = AdamW(learning_rate=lr_schedule)

# model.compile(loss={'output1': 'mean_squared_error', 'output2': 'mean_squared_error'}, optimizer="Adam", learning_rate=lr_schedule)
model.compile(loss={'output1': 'mean_squared_error', 'output2': 'mean_squared_error'}, optimizer = keras.optimizers.AdamW(learning_rate=lr_schedule))

# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# Fit the model
history = model.fit(X_train, {'output1': Y_train[0], 'output2': Y_train[1]},
                    validation_data=(X_val, {'output1': Y_val[0], 'output2': Y_val[1]}),
                    epochs=10000000, batch_size=10, verbose=2, callbacks=[es])

# Calculate predictions
PredTrainSet = model.predict(X_train)
PredValSet = model.predict(X_val)

# Save predictions
# Note: This will give you two arrays for each set of predictions, handle each according to your needs
np.savetxt("/home/deamoon_uw_nn/bucket_source/T_results_output1.csv", PredTrainSet[0], delimiter=",")
np.savetxt("/home/deamoon_uw_nn/bucket_source/T_results_output2.csv", PredTrainSet[1], delimiter=",")
np.savetxt("/home/deamoon_uw_nn/bucket_source/V_results_output1.csv", PredValSet[0], delimiter=",")
np.savetxt("/home/deamoon_uw_nn/bucket_source/V_results_output2.csv", PredValSet[1], delimiter=",")

model.save('/home/deamoon_uw_nn/bucket_source/uw_nn.h5')
# Model.save('/home/deamoon_uw_nn/bucket_source/uw_nn.keras')
