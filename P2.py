import numpy as np 
import math
from keras.datasets import mnist
from utils import *
from data import *

'''Hyper parameters
'''
EPOCH = 1
BATCH_SIZE = 100
LEARNINT_RATE = 0.1
NUM_OF_CLASSIFIERS = 10

'''Training
'''
classifiers = []
for i in range(NUM_OF_CLASSIFIERS): # We need to train 10 models for each i
    parameter = init_parameters([784, 1]) # 1 * (1, 784) list of ndarray
    for epoch in range(EPOCH):      # num of epochs we want, 1 in our case
        for batch, X_t in enumerate(x_train_batch):
            output = sigmoid(X_t, parameter[0]) #(1, 100)
            cost, dw = corss_entropy_cost(X_t, output, y_train_batch_sets[i][batch])
            parameter[0] -= LEARNINT_RATE * dw.T #updata parameter
            if not batch % 100: #print cost in eatch 100 step
                print("model %d cost: " % i + str(cost))
    classifiers.append(parameter)

''' Predicting
'''

print("First ten test label: ")
print((y_test_flatten[0][:40]))
print("First ten predict label: ")
print(predict(classifiers, x_test_flatten[:, :40]))

test_output = predict(classifiers, x_test_flatten)
accuracy = np.sum(np.equal(y_test_flatten[0], test_output[0])) / len(y_test_flatten[0])
print("Accuracy: " + str(accuracy))