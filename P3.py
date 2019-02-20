import numpy as np 
import math
from keras.datasets import mnist
from utils import *
from data import *

'''Hyper parameters
'''
EPOCH = 10
BATCH_SIZE = 50
LEARNINT_RATE = 0.01
NUM_OF_CLASSIFIERS = 10

'''Training
'''
parameters = init_parameters([784, 10])
W = parameters[0]
# print(W.shape)
bias = 0
for epoch in range(EPOCH):
    for batch, X_t in enumerate(x_train_batch):
        Z = np.dot(W, X_t) + bias
        A = softmax(Z)
        y = y_train_onehot_batch[batch]
        dW = np.dot(X_t, (A - y).T) / BATCH_SIZE
        db = np.sum((A - y) / BATCH_SIZE)
        W -= LEARNINT_RATE * dW.T
        bias -= LEARNINT_RATE * db
        # if not batch % 100:
        #     print("cost: " + str(cost))
    Y_test_predict = np.argmax(softmax(np.dot(W, x_test_flatten) + bias), axis=0).reshape(1, 10000)
    accuracy = np.mean(Y_test_predict == y_test_flatten)
    print("Epoch %d:" % epoch)
    print("Accuracy: " + str(accuracy) + '\n')

''' Predicting
'''
Y_test_predict = np.argmax(softmax(np.dot(W, x_test_flatten) + bias), axis=0).reshape(1, 10000)
accuracy = np.mean(Y_test_predict == y_test_flatten)
print("\nFinal Accuracy: " + str(accuracy))
print("First 10 labels: ")
print(y_test_flatten[:10])
print("First 10 prediction: ")
print(Y_test_predict[:10])
