import numpy as np 
import math
from keras.datasets import mnist
from utils import *

'''Hyper parameters
'''
BATCH_SIZE = 100
NUM_OF_CLASSIFIERS = 10

'''Data preparation
'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()  #x_train(60000, 28, 28), y_train(60000), x_test(10000, 28, 28), y_test(10000,)  datatype: ndarrays

#reshaping training ans test examples
x_train_flatten = x_train.reshape(x_train.shape[0], -1).T /255   #(784, 60000)
y_train_flatten = y_train.reshape(-1, y_train.shape[0])  #(1, 60000)
x_test_flatten = x_test.reshape(x_test.shape[0], -1).T / 255     #(784, 10000)
y_test_flatten = y_test.reshape(-1, y_test.shape[0])        #(1, 10000)

#one_hot represent of Y
y_train_onehot = one_hot(y_train_flatten) # (10, 60000)
y_test_onehot = one_hot(y_test_flatten)   # (10, 10000)
# print(y_train_onehot.shape)

#Init x_batch ans y_batch
x_train_batch = init_batch(x_train_flatten, batch_size=BATCH_SIZE) # (600, 784, 100)
x_test_batch = init_batch(x_test_flatten, batch_size=BATCH_SIZE)   # (100, 784, 100)
y_train_onehot_batch = init_batch(y_train_onehot, batch_size=BATCH_SIZE) #(600, 10, 100)
# print(y_train_flatten[0][:3])
# print(y_trian_onehot_batch[0][:, :3])

#set y set to i representation
y_train_sets = [] #10 sets for 10 models
for i in range(10):
    y_train_sets.append(np.asarray([[1 if num == i else 0 for num in y_train_flatten[0]]]))

y_test_sets = []
for i in range(10):
    y_test_sets.append(np.asarray([[1 if num == i else 0 for num in y_test_flatten[0]]]))

#batch sets for 10 y sets
y_train_batch_sets = []   # (600, 1, 100) * 10
for model in range(10):
    y_train_batch_sets.append(init_batch(y_train_sets[model], batch_size=BATCH_SIZE))

y_test_batch_sets = []    # (100, 1, 100) * 10
for model in range(10):
    y_test_batch_sets.append(init_batch(y_test_sets[model], batch_size=BATCH_SIZE))

