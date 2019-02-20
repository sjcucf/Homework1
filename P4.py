import numpy as np 
import math
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

#data preparation
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28 * 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28 * 28).astype('float32')

#normalization
X_train /= 255
X_test /= 255

#y to one-hot
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

def model():
    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(28 * 28,)))
    #compile model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#Setup model
model = model()
model.fit(X_train, Y_train, epochs=10, batch_size=100, validation_data=(X_test, Y_test))
