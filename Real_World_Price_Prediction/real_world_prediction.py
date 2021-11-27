#!/usr/bin/env python3

# Import required libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten

# Read Dataset
train_df = pd.read_csv('historical_coin_data_5m.csv')
print(np.shape(train_df))
test_df = pd.read_csv('test_data.csv')
print(np.shape(test_df))

# Choose a currency
currency = 'Stellar'

# Originally the data has a 5-minute interval
# We can use Python to get hourly interval using this syntax: [::12] because 60 / 5 = 12
# Skip every 12 values
train_price = train_df[train_df['Currency'] == currency]['Price USD'][::12]
train_vol = train_df[train_df['Currency'] == currency]['Trading Volume Last 24h'][::12]
train_marketcap = train_df[train_df['Currency'] == currency]['Market Cap'][::12]
train_datetime = train_df[train_df['Currency'] == currency]['DateTime'][::12]

# From the above chart, the number range is too large (**1e10**), so we will use the log scale instead.
train_feats = np.dstack([np.log(train_price[1:] + 1), np.log(train_vol[1:] + 1), np.log(train_marketcap[1:] + 1)])[0,:,:]

# Data scaling
scaler = RobustScaler()  # Robust scaler: Helps removing outliers
scaler2 = MinMaxScaler()  # Min-max normalisation
train_feats_scaled = scaler2.fit_transform(scaler.fit_transform(train_feats))

# Determine best sliding window size
sliding_window = np.lib.stride_tricks.sliding_window_view(train_feats_scaled, (5 * 7 + 3 ,3), axis=(0, 1))
sliding_window = sliding_window.reshape((sliding_window.shape[0], sliding_window.shape[2], sliding_window.shape[3]))

# Xs: Price, volume, market cap and price of the past week
x_train = np.array([i[:-3] for i in sliding_window])
# Xs: Price of the next three hours
y_train = np.array([i[-3:, 0] for i in sliding_window])
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))

# Originally the data has a 5-minute interval
# We can use Python to get hourly interval using this syntax: [::12] because 60 / 5 = 12
# Skip every 12 values
test_price = test_df[test_df['Currency'] == currency]['Price USD'][::12]
test_vol = test_df[test_df['Currency'] == currency]['Trading Volume Last 24h'][::12]
test_marketcap = test_df[test_df['Currency'] == currency]['Market Cap'][::12]
test_datetime = test_df[test_df['Currency'] == currency]['DateTime'][::12]

# From the above chart, the number range is too large (**1e10**), so we will use the log scale instead.
test_feats = np.dstack([np.log(test_price[1:] + 1), np.log(test_vol[1:] + 1), np.log(test_marketcap[1:] + 1)])[0,:,:]

# Data scaling
scaler = RobustScaler()  # Robust scaler: Helps removing outliers
scaler2 = MinMaxScaler()  # Min-max normalisation
test_feats_scaled = scaler2.fit_transform(scaler.fit_transform(test_feats))

# Determine best sliding window size
sliding_window = np.lib.stride_tricks.sliding_window_view(test_feats_scaled, (5 * 7 + 3 ,3), axis=(0, 1))
sliding_window = sliding_window.reshape((sliding_window.shape[0], sliding_window.shape[2], sliding_window.shape[3]))

# Xs: Price, volume, market cap and price of the past week
x_test = np.array([i[:-3] for i in sliding_window])
# Xs: Price of the next three hours
y_test = np.array([i[-3:, 0] for i in sliding_window])
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

############################################################################################################


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

start = time.time()

# LSTM Model
model = Sequential()
model.add(LSTM(128, dropout=0.05, input_shape=(5 * 7, 3), return_sequences=True))
model.add(LSTM(64, dropout=0.05))
model.add(Dense(3, activation='sigmoid'))

# MLP Model
#model = Sequential()
#model.add(Flatten(input_shape=(5 * 7, 3)))
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.1))
#model.add(Dense(3, activation='sigmoid'))

# Adam optimiser allows high learning rate at first and speeds up training
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()])

# Early stop to prevent overfitting
callbackEs = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1, restore_best_weights=True)

# Log performance using TensorBoard
callbackTb = tf.keras.callbacks.TensorBoard()

# Fit the model with the training set and output MSE, RMSE
model.fit(x_train, y_train, shuffle=True, epochs=50, verbose=2, validation_data=(x_test, y_test), callbacks = [callbackTb, callbackEs])

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
plt.savefig('LSTM-final-predict.png')
