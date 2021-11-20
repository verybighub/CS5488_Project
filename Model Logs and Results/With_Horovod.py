#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import useful packages
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# Import required libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import time


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten

import tensorflow as tf
import horovod.tensorflow.keras as hvd

# Read Dataset
df = pd.read_csv('historical_coin_data_5m.csv')

# Choose a currency
currency = 'Stellar'

# Originally the data has a 5-minute interval
# We can use Python to get hourly interval using this syntax: [::12] because 60 / 5 = 12
# Skip every 12 values
price = df[df['Currency'] == currency]['Price USD'][::12]
vol =  df[df['Currency'] == currency]['Trading Volume Last 24h'][::12]
marketcap =  df[df['Currency'] == currency]['Market Cap'][::12]
datetime =  df[df['Currency'] == currency]['DateTime'][::12]

# From the above chart, the number range is too large (**1e10**), so we will use the log scale instead.
feats=np.dstack([np.log(price[1:] + 1),np.log(vol[1:] + 1),np.log(marketcap[1:] + 1)])[0,:,:]

# Data scaling
# Robust scaler: Helps removing outliers
scaler = RobustScaler()
# Min-max normalisation: Scale all features between 0-1: model performs better when numerical input variables are scaled to a standard range
scaler2 = MinMaxScaler()
feats_scaled = scaler2.fit_transform(scaler.fit_transform(feats))

# Determine best sliding window size
# size: 3 * 7 => last three days
sliding_window = np.lib.stride_tricks.sliding_window_view(feats_scaled, (5 * 7 + 3 ,3), axis=(0,1))
sliding_window = sliding_window.reshape((sliding_window.shape[0], sliding_window.shape[2], sliding_window.shape[3]))

# Xs: Price, volume, market cap and price of the past week
Xs = np.array([i[:-3] for i in sliding_window])
# Xs: Price of the next three hours
Ys = np.array([i[-3:, 0] for i in sliding_window])

# Data Spliting
TrainLen = int(len(Xs) * 0.90)
ValLen = int(len(Xs) * 0.95)
TestLen = len(Xs) - TrainLen

x_train = Xs[0:TrainLen,:]
y_train = Ys[0:TrainLen]
x_val = Xs[TrainLen:ValLen,:]
y_val = Ys[TrainLen:ValLen]
x_test = Xs[ValLen:,:]
y_test = Ys[ValLen:]

y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
y_val = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

#print(x_train.shape)
#print(y_train.shape)
#print(x_val.shape)
#print(y_val.shape)
#print(x_test.shape)
#print(y_test.shape)

start = time.time()

# Horovod: initialize Horovod.
hvd.init()

# LSTM Model
#model = Sequential()
#model.add(LSTM(128, dropout=0.05, input_shape=(5 * 7, 3), return_sequences=True))
#model.add(LSTM(64, dropout=0.05))
#model.add(Dense(3, activation='sigmoid'))

# MLP Model
model = Sequential()
model.add(Flatten(input_shape=(5 * 7, 3)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(3, activation='sigmoid'))

# Horovod: adjust learning rate based on number of GPUs.
scaled_lr = 0.001 * hvd.size()
opt = tf.optimizers.Adam(scaled_lr)

# Horovod: add Horovod DistributedOptimizer.
opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

# uses hvd.DistributedOptimizer() to compute gradients.
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error', 
	'mean_squared_error',
	tf.keras.metrics.RootMeanSquaredError()], 
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
	# Early stop to prevent overfitting
	callbackEs = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)
	
	# Log performance using TensorBoard
	callbackTb = tf.keras.callbacks.TensorBoard()
	callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Horovod: write logs on worker 0.
verbose = 1 if hvd.rank() == 0 else 0

# Train the model.
# Horovod: adjust number of steps based on number of GPUs.
model.fit(x_train, 
	y_train, 
	shuffle=True, 
	steps_per_epoch=500 // hvd.size(), 
	validation_data=(x_val, y_val), 
	callbacks=[callbacks, callbackTb, callbackEs], 
	epochs=50, 
	verbose=verbose)

end = time.time()
print('Used training time: %f' %(end - start))

# Prediction on the testing set
model_pre = model.predict(x_test)
test_predicted = np.ravel(model_pre)
test_label = np.ravel(y_test)
mse = np.mean((test_predicted - test_label) ** 2)
rmse = np.sqrt(np.mean((test_predicted - test_label) ** 2))
mae = np.mean(np.abs(test_predicted - test_label))
print('Testing set: MSE %f, RMSE: %f, MAE: %f' % (mse, rmse, mae))

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
	fig, ax = plt.subplots(1, figsize=(16, 9))
	ax.plot(line1, label=label1, linewidth=lw)
	ax.plot(line2, label=label2, linewidth=lw)
	ax.set_ylabel('price [USD]', fontsize=14)
	ax.set_title(title, fontsize=18)
	ax.legend(loc='best', fontsize=18)
	
line_plot(test_label, test_predicted, 'actual', 'prediction', lw=3)
plt.savefig('MLP-with-Horovod.png')
