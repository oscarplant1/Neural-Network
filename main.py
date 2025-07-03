#%%
import math
import numpy as np 
import scipy as sp 
import pandas as pd
import random
import csv
from matplotlib import pyplot as plt
import Network
import pickle

# Add noise to the images
def add_noise(X, noise_level=0.2):
    noisy = X + noise_level * np.random.randn(*X.shape)
    noisy = np.clip(noisy, 0., 1.)
    return noisy

# Rotate and shift images
def augment_image(img, max_shift=2, max_angle=15):
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    shifted = sp.ndimage.shift(img, shift=(shift_x, shift_y), mode='constant', cval=0.0)
    
    angle = np.random.uniform(-max_angle, max_angle)
    rotated = sp.ndimage.rotate(shifted, angle, reshape=False, mode='constant', cval=0.0)
    return rotated

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


### Add some noise, shifts and rotations to improve the networks accuracy on data not from the training or test sets

# Add some random noise
X_train = add_noise(X_train, noise_level=0.2)

# Shift and rotate each image
X_train_aug = np.zeros_like(X_train)
for i in range(X_train.shape[1]):
    img = X_train[:, i].reshape(28, 28)
    img_aug = augment_image(img)
    X_train_aug[:, i] = img_aug.flatten()
X_train = X_train_aug

def one_hot(Y, num_classes):
    one_hot_Y = np.zeros((num_classes, Y.size))
    one_hot_Y[Y.astype(int), np.arange(Y.size)] = 1
    return one_hot_Y

# Onehot all the labels
Y_test = one_hot(Y_test,10)
Y_train = one_hot(Y_train,10)


np.set_printoptions(suppress=True)

# This configuration gives an accuarcy of 94.9% on the test data 
Neural_Network = Network.Network([784,400,100,10])
Neural_Network.trainNetwork(X_train,Y_train,0.1,32,50,True,X_test,Y_test)

# Save trained network
with open("trained_network.pkl", "wb") as f:
    pickle.dump(Neural_Network, f)
# %%
