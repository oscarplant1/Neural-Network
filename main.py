#%%
import math
import numpy as np 
import scipy as sp 
import pandas as pd
import random
import csv
from matplotlib import pyplot as plt
import Network

#60000 images
training_data = pd.read_csv('./MNIST_CSV/mnist_train.csv')
#10000 images
test_data = pd.read_csv('./MNIST_CSV/mnist_test.csv')

training_data = np.array(training_data)
test_data = np.array(test_data)

# Shuffle data
m, n = training_data.shape
p, q = test_data.shape
np.random.shuffle(training_data)
np.random.shuffle(test_data)

# Splitting labels from test data
data_test = test_data[0:p].T
Y_test = data_test[0]
X_test = data_test[1:q]
X_test = X_test / 255.

# Splitting labels from training
data_train = training_data[0:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.

def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y.astype(int), np.arange(Y.size)] = 1
    return one_hot_Y

# Onehot all the labels
Y_test = one_hot(Y_test,10)
Y_train = one_hot(Y_train,10)


np.set_printoptions(suppress=True)

Neural_Network = Network.Network([784,400,100,10])
Neural_Network.trainNetwork(X_train,Y_train,0.015,64,50,True,X_test,Y_test)
# %%
