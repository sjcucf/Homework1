import numpy as np 
from utils import *
import queue
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import models
from keras import layers

'''data preparation
'''
(X_train_ori, Y_train_ori), (X_test_ori, Y_test_ori) = mnist.load_data()
X_train = X_train_ori.reshape(X_train_ori.shape[0], 28 * 28).astype('float32')
X_test = X_test_ori.reshape(X_test_ori.shape[0], 28 * 28).astype('float32')

#normalization
X_train /= 255
X_test /= 255

#y to one-hot
Y_train = np_utils.to_categorical(Y_train_ori)
Y_test = np_utils.to_categorical(Y_test_ori)

''' Compute connect component
'''
# Find white regions in images using BFS
directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, 1), (1, -1), (-1, -1))
def compute_regions(images):
    regions = []
    for index, image in enumerate(images):
        print(index)
        region = 0
        copy = np.copy(image)
        # print(copy.shape)
        for i in range(copy.shape[0]):
            for j in range(copy.shape[1]):
                q = []
                if copy[i][j] == 0:
                    region += 1
                    q.append((i, j))
                    while (q):
                        x_curr, y_curr = q.pop(0)
                        # copy[x_curr][y_curr] = 1
                        for d in directions:
                            x_next = x_curr + d[0]
                            y_next = y_curr + d[1]
                            if (x_next >= 0 and x_next < copy.shape[0] and y_next >= 0 and y_next < copy.shape[1] and copy[x_next][y_next] == 0):
                                q.append((x_next, y_next))
                                copy[x_next][y_next] = 1
        regions.append(region) 
    return np.asarray([regions])

# print(Y_train_ori[:30])
train_regions = compute_regions(X_train_ori[:3000]).T
test_regions = compute_regions(X_test_ori[:3000]).T
# print(train_regions.shape)
X_train_new = np.concatenate((X_train[:3000], train_regions), axis=1)
X_test_new = np.concatenate((X_test[:3000], test_regions), axis=1)
# print(X_train_new.shape)

def model():
    model = Sequential()
    model.add(Dense(10, activation='softmax', input_shape=(28 * 28 + 1,)))
    #compile model
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#Setup model
model = model()
model.fit(X_train_new[:3000], Y_train[:3000], epochs=10, batch_size=100, validation_data=(X_test_new[:3000], Y_test[:3000]))












                                

                
                






