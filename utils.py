import numpy as np 
import math
from keras.datasets import mnist

#convert the input data into batchs
def init_batch(inputs, batch_size):
    num_full_batch = len(inputs[0]) // batch_size
    batchs = []
    for i in range(num_full_batch):
        batchs.append(inputs[:,i * batch_size : (i+1) * batch_size])
    if len(inputs[0]) % batch_size:
        batchs.append(inputs[:, num_full_batch * batch_size :])
    return np.asarray(batchs)
# print(init_batch(x_train_flatten, BATCH_SIZE))

#Initialize parameters
def init_parameters(layers):
    ''' Initilize the parameters of each layer
        In this case, we only have input layer(28*28) and output layer(1) 
    '''
    parameters = []
    for layer in range(1, len(layers)):
        # print(len(layers))
        # print(layers[layer])
        parameters.append(np.zeros((layers[layer], layers[layer - 1])))
        # print(parameters[0].shape)
    return parameters

# parameters = init_parameters(layers)
# print(len(parameters))

#forward propagation using only logistic regression function
def sigmoid(X, w):
    z = np.dot(w, X)
    # print(z[0][:2])
    res = 1 / (1 + np.exp(-z))
    # print(res[0][:5])
    # samples = [i for i in res[0] if i == 0]
    # print(samples)
    return res 

# print(sigmoid(x_train_flatten[:, 1], parameters[0]))

def mean_square_cost(X, A, Y):
    m = Y.shape[1] 
    cost = (1 / m) * np.sum(np.square(A - Y))
    dw = (1 / m) * np.dot(X, (2 * (A - Y) * A * (1 - A)).T)
    return cost, dw
    
def corss_entropy_cost(X, A, Y):
    m = Y.shape[1]
    # print(m)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    return cost, dw

def predict(classifiers, X):
    outputs = []
    for parameter in classifiers:
        outputs.append(sigmoid(X, parameter[0]))
    return np.argmax(outputs, axis=0)

def one_hot(Y): # (1, 60000) to (10, 60000)
    res = []
    for num in Y[0]:
        entry = np.zeros(10)
        entry[num] = 1
        res.append(entry)
    res = np.asarray(res)
    return res.T

def softmax(Z): # (10, num_of_examples)
    exps = np.exp(Z)
    return exps / np.sum(exps, axis=0) #compute softmax based on rows

def multiclass_cross_entropy(Z, Y):
    Y = Y.T 
    Z = Z.T
    m = Y.shape[0]
    p = softmax(Z)
    log_likelihood = -np.log(p[range(m), Y])
    loss = np.sum(log_likelihood) / m
    return loss

    
# Y = np.asarray([[1,2,3]])
# res = one_hot(Y)
# print(res)
# x = np.random.randn(3, 5)
# print(softmax(x))

        
        









    




