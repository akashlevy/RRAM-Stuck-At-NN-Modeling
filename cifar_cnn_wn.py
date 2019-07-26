'''
CIFAR-10 example from https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
Now with weight normalization. Lines 64 and 69 contain the changes w.r.t. original.
'''

from __future__ import print_function

import tensorflow as tf
with tf.Session() as sess:
    devices = sess.list_devices()

from keras.datasets import cifar10, cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

dataset = cifar100

batch_size = 32
nb_classes = 100
nb_epoch = 200
data_augmentation = False

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = dataset.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original). EDIT: now with weight normalization, so slightly more original ;-)
from weightnorm import SGDWithWeightnorm
sgd_wn = SGDWithWeightnorm(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd_wn,metrics=['accuracy'])

# data based initialization of parameters
from weightnorm import data_based_init
data_based_init(model, X_train[:100])

model_path = 'models/cifar_cnn.%s.h5' % dataset.__name__

if not os.path.exists(model_path):
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                validation_data=(X_test, Y_test),
                shuffle=True)
    else:
        print('Using real-time data augmentation.')

        # this will do preprocessing and realtime data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen.fit(X_train)

        # fit the model on the batches generated by datagen.flow()
        model.fit_generator(datagen.flow(X_train, Y_train,
                            batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test))

    model.save_weights(model_path)
else:
    model.load_weights(model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

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
