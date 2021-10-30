#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import useful packages
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Hide the Configuration and Warnings
import math
import pandas as pd
import numpy as np

import json
import requests

import horovod.tensorflow.keras as hvd

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Read all the data
total_data = pd.read_csv('Bitcoin.csv')
total_data = pd.DataFrame(total_data, columns=['Price', 'Volume', 'Market'])
total_data = np.array(total_data).astype('float64')
# print('Total number of data is ', np.shape(total_data)[0], '\n')

# Get the features (Volume and Market Cap)
data = total_data[:, 1:]
labels = total_data[:, 0] / 1000.0
print(np.shape(data), np.shape(labels), '\n')

# Split the data into training set and testing set
train_test_ratio = 0.8
num_data = np.shape(data)[0]
train_data = data[:int(train_test_ratio*num_data), :]
train_label = labels[:int(train_test_ratio*num_data)]
train_label = np.reshape(train_label, [-1, 1])

test_data = data[int(train_test_ratio*num_data):, :]
test_label = labels[int(train_test_ratio*num_data):]
test_label = np.reshape(test_label, [-1, 1])

# Data Normalization
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)
test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)

# Data Prepare for the LSTM Model
n_input = 2    # The input size of signals at each time
max_time = 64   # The unfolded time slices of the LSTM Model

num_train_max_time = int(np.shape(train_data)[0] / max_time)
train_data = np.reshape(train_data[:num_train_max_time * max_time, :], [-1, max_time, n_input])
train_label = np.mean(np.reshape(train_label[:num_train_max_time * max_time, :], [-1, max_time]), axis=1, keepdims=True)

num_test_max_time = int(np.shape(test_data)[0] / max_time)
test_data = np.reshape(test_data[:num_test_max_time * max_time, :], [-1, max_time, n_input])
test_label = np.mean(np.reshape(test_label[:num_test_max_time * max_time, :], [-1, max_time]), axis=1, keepdims=True)

print('Number of training data is ', np.shape(train_data))
print('Number of training labels is ', np.shape(train_label))
print('Number of testing data is ', np.shape(test_data))
print('Number of testing labels is ', np.shape(test_label), '\n\n')

train_data = tf.convert_to_tensor(train_data)
train_label = tf.convert_to_tensor(train_label)

test_data = tf.convert_to_tensor(test_data)
test_label = tf.convert_to_tensor(test_label)

#########################################################################

# Model Hyper-parameters
lstm_neurons = 128
epochs = 10
batch_size = 4
loss = 'mae'
dropout = 0.25
optimizer = 'adam'
output_size = 1

# Horovod: initialize Horovod.
hvd.init()

model = tf.keras.models.Sequential([
    LSTM(lstm_neurons, input_shape=(max_time, n_input)),
    Dropout(dropout),

    # tanh
    Dense(units=512, activation='relu'),
    Dropout(dropout),

    Dense(units=256, activation='relu'),
    Dropout(dropout),

    Dense(units=128, activation='relu'),
    Dropout(dropout),

    Dense(units=64, activation='relu'),
    Dropout(dropout),

    Dense(units=32, activation='relu'),
    Dropout(dropout),

    Dense(units=output_size, activation='linear'),


])

# model.add(Dense(1))


# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

# Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss=loss,
                optimizer=opt,
                metrics=['mae', 'mse'],
                experimental_run_tf_function=False)

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    
    # Horovod: average metrics among workers at the end of every epoch.
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),
    
    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
    hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
model.fit(train_data, train_label, steps_per_epoch=500 // hvd.size(), callbacks=callbacks, epochs=epochs, verbose=verbose)

# model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
#model.evaluate(test_data, test_label, verbose=verbose)
targets = test_label
preds = model.predict(test_data).squeeze()
#preds = test_data['Price'].values[:-max_time] * (preds + 1)
#preds = pd.Series(data=preds)

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)

line_plot(targets,preds,'actual','prediction',lw=3)

plt.savefig('result.png')



