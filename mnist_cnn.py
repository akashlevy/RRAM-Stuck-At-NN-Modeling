'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function

import os
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"

import keras
from keras.datasets import mnist, fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

batch_size = 128
num_classes = 10
epochs = 12

dataset = fashion_mnist

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = dataset.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model_path = "models/mnist_cnn.%s.h5" % dataset.__name__
if not os.path.exists(model_path):
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    model.save_weights(model_path)
else:
    model.load_weights(model_path)
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# QUANTIZATION ADDON

#LEVELS = 3
#MIN = -0.1
#MAX = 0.1

LEVELS = 4
MIN = -0.3
MAX = 0.3

def quantize(weights, levels, min_range, max_range):
    range_size = max_range - min_range
    qweights = np.round((weights - min_range) * (levels-1) / range_size) * range_size / (levels-1) + min_range
    #print('Weights', weights)
    #print('QWeights', qweights)
    return np.clip(qweights, min_range, max_range)

model_weights = model.get_weights()
#plt.hist(np.concatenate([weights.flatten() for weights in model_weights]),bins=16,range=(-0.5,0.5), log=True)
#plt.show()
quantized_weights = [quantize(weights, LEVELS, MIN, MAX) for weights in model_weights]
model.set_weights(quantized_weights)
score = model.evaluate(x_test, y_test, verbose=0)
print('Quantized test loss:', score[0])
print('Quantized test accuracy:', score[1])

# QUANTIZED NAIVE BER MODELING
BER = 0.0156

ber_weights = [np.where(np.random.random(weights.shape) < BER * LEVELS/(LEVELS-1), np.random.choice(np.linspace(MIN, MAX, LEVELS)), weights) for weights in quantized_weights]
model.set_weights(ber_weights)
score = model.evaluate(x_test, y_test, verbose=0)
print('Naive BER test loss:', score[0])
print('Naive BER test accuracy:', score[1])

# EXACT MODELING
LOWER_BOUND_PROBS = np.array([253., 4., 2., 5.])
MASKED = LOWER_BOUND_PROBS * np.array([1, 1, 0, 0])
LOWER_BOUND_PROBS += 253 * MASKED / MASKED.sum()
MASKED = LOWER_BOUND_PROBS * np.array([1, 1, 1, 0])
LOWER_BOUND_PROBS += 253 * MASKED / MASKED.sum()
#print(LOWER_BOUND_PROBS)
LOWER_BOUND_PROBS /= LOWER_BOUND_PROBS.sum()

UPPER_BOUND_PROBS = np.array([3., 1., 1., 259.])
MASKED = UPPER_BOUND_PROBS * np.array([0, 0, 1, 1])
UPPER_BOUND_PROBS += 254 * MASKED / MASKED.sum()
MASKED = UPPER_BOUND_PROBS * np.array([0, 1, 1, 1])
UPPER_BOUND_PROBS += 256 * MASKED / MASKED.sum()
#print(UPPER_BOUND_PROBS)
UPPER_BOUND_PROBS /= UPPER_BOUND_PROBS.sum()

lower_bounds = [np.random.choice(np.linspace(MIN, MAX, LEVELS), size=weights.shape, p=LOWER_BOUND_PROBS) for weights in quantized_weights]
upper_bounds = [np.random.choice(np.linspace(MIN, MAX, LEVELS), size=weights.shape, p=UPPER_BOUND_PROBS) for weights in quantized_weights]
exact_weights = [np.clip(weights, lower_bound, upper_bound) for weights, lower_bound, upper_bound in zip(quantized_weights, lower_bounds, upper_bounds)]
model.set_weights(exact_weights)
score = model.evaluate(x_test, y_test, verbose=0)
print('Exact modeling test loss:', score[0])
print('Exact modeling test accuracy:', score[1])