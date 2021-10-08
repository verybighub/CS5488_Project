#################################
#
#	Draft only
#
#################################



import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.preprocessing import MinMaxScaler

#import horovod.tensorflow as hvd

DataPCh = Data.pct_change()		# pandas pct_change

LogReturns = np.log(1 + DataPCh)	# Logarithm of returns

print(LogReturns.tail(10))
plt.figure(figsize=(10,5))
plt.plot(LogReturns)
plt.show()

# Data scaling
scaler = MinMaxScaler()
DataScaled = scaler.fit_transform(Data)

# Data splitting
TrainLen = int(len(DataScaled) * 0.70)
TestLen = len(DataScaled) - TrainLen
TrainData = DataScaled[0:TrainLen,:]
TestData = DataScaled[TrainLen:len(DataScaled),:]

print(len(TrainData), len(TestData))

TrainX = np.reshape(TrainX, (TrainX.shape[0], 1, TrainX.shape[1]))
TestX = np.reshape(TestX, (TestX.shape[0], 1, TestX.shape[1]))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(LSTM(256, input_shape=(1, TimeStep)))
model.add(Dense(1, activation='sigmoid'))

##########################################################################

from tensorflow.keras import backend as K
#from tensorflow.keras.models import Sequential
#import tensorflow as tf
from tensorflow import keras
import horovod.tensorflow.keras as hvd

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
	tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
(x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())
model = get_model(num_classes)
optimizer = keras.optimizers.Adadelta(lr=learning_rate * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]
if hvd.rank() == 0:
	callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True))
model.fit(TrainX, TrainY, batch_size=1.0, callbacks=callbacks, epochs=100, verbose=2, validation_data=(x_test, y_test))

model.summary()

##########################################################################

score = model.evaluate(TrainX, TrainY, verbose=0)
print('Keras Model Loss = ',score[0])
print('Keras Model Accuracy= ',score[1])

TrainPred = model.predict(TrainX)
TestPred = model.predict(TestX)

TrainPred = scaler.inverse_transform(TrainPred)
TrainY = scaler.inverse_transform([TrainY])
TestPred = scaler.inverse_transform(TestPred)
TestY = scaler.inverse_transform([TestY])

TrainPredictPlot = np.empty_like(DataScaled)
TrainPredictPlot[:, :] = np.nan
TrainPredictPlot[1:len(TrainPred)+1, :] - TrainPred

TestPredictPlot = np.empty_like(DataScaled)
TestPredictPlot[:, :] = np.nan
TestPredictPlot[len(TrainPred)+(1*2)+1:len(DataScaled)-1, :] = TestPred

plt.plot(scaler.inverse_transform(DataScaled))
plt.plot(TrainPredictPlot)
plt.plot(TestPredictPlot)
plt.show()


def get_dataset(num_classes, rank=0, size=1):
	from tensorflow import keras
	(x_train, y_train), (x_test, y_test) = keras.datasets.
	mnist.load_data('MNIST-data-%d' % rank)
	x_train, y_train = \
	x_train[rank:size],y_train[rank:size]
	x_test,y_test = x_test[rank:size],y_test[rank:size]
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	x_train = x_train.astype('float32')/255
	x_test = x_test.astype('float32')/255
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	return (x_train, y_train), (x_test, y_test)

'''
def train_hvd(checkpoint_path, learning_rate=1.0):
	from tensorflow.keras import backend as K
	from tensorflow.keras.models import Sequential
	import tensorflow as tf
	from tensorflow import keras
	import horovod.tensorflow.keras as hvd

	hvd.init()
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	if gpus:
		tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
	(x_train, y_train), (x_test, y_test) = get_dataset(num_classes, hvd.rank(), hvd.size())
	model = get_model(num_classes)
	optimizer = keras.optimizers.Adadelta(lr=learning_rate * hvd.size())
	optimizer = hvd.DistributedOptimizer(optimizer)
	model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),]
	if hvd.rank() == 0:
		callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True))
	model.fit(x_train, y_train, batch_size=batch_size, callbacks=callbacks, epochs=epochs, verbose=2, validation_data=(x_test, y_test))
'''

'''
hvd.init()

config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())

opt = tf.train.AdagradOptimizer(0.01 * hvd.size())

opt = hvd.DistributedOptimizer(opt)
'''

'''
strategy = tf.distribute.MirroredStrategy()

model.load_weights(tf.train.liatest_checkpoint(checkpoint_dir))

eval_loss, eval_acc = model.evaluate(test_dataset)
print('Eval loss: {}, Eval Accuracy: {}' .format(eval_loss, eval_acc))
'''

'''
from sparkdt import HorovodRunner
hr = HorovodRunner(np=2)
hr.run(train_hvd, checkpoint_path=CHECKPOINT_PATH,
tearning_rate=LEARNING_RATE)
'''