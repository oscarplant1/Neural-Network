import math
import numpy as np 
import scipy as sp 
import pandas as pd
import random
import csv
from matplotlib import pyplot as plt
import Layer

# Neural class using ReLU as the hidden layer activation function, softmax as the final layer activation function and categorical cross entropy loss as the cost/loss function

class Network:
    def __init__(self, nodesPerLayer):
        self.nodesPerLayer = nodesPerLayer

        # Initialise list of layers
        self.layers = []
        for i in range(len(self.nodesPerLayer)-1):
            self.layers.append(Layer.Layer(nodesPerLayer[i],nodesPerLayer[i+1]))
    
    # Using cross-entropy loss for the cost function of the network - not used yet
    def costFunction(self,inputs,expectedResults):
        # Add small epsilon to avoid log(0)
        epsilon = 1e-12
        Cost = -np.sum(expectedResults * np.log(inputs + epsilon)) / inputs.shape[1]
        return Cost
    
    def costFunctionDerivative(self,inputs,expectedResults):
        return inputs - expectedResults

    def forwardPass(self,inputs):
        # Reset the inputs and weighted inputs stores for each layer
        for layer in self.layers:
            layer.inputs = []
            layer.weightedInputs = []

        # Feed the inputs forward through every layer of the network but the last
        for i in range(len(self.layers)-1):
            inputs = self.layers[i].calculateOutputs(inputs)[1]

        # Compute the output of the final layer using the different activation function
        inputs = self.layers[-1].calculateOutputLayer(inputs)[1]
        return inputs
    
    # Back propogation
    def backwardPass(self,finalLayerActivationValues,expectedValues):
        # Compute the error of the forward pass 
        OutputError = self.costFunctionDerivative(finalLayerActivationValues,expectedValues)

        # Derivative of the cost with respect to the weighted input (z) is the same as the derivative with respect     to the activated output (a)
        dCdz = OutputError

        # Loop backwards through the layer list, calculating gradients and updating each layer with these gradients
        for i in reversed(range(len(self.layers))):
            current_layer = self.layers[i]
            z = current_layer.weightedInputs[-1]
            previousActivations = current_layer.inputs[-1]

            # Calculate final layer seperatley
            if i == len(self.layers) -1:
                current_layer.computeGradients(dCdz,previousActivations)
            else:
                next_layer = self.layers[i+1]
                next_weights = next_layer.weightsIn

                # dC/dz = W_(L+1)^T * dC/dz_(L+1) * ReLU'(z_L) 
                dCdz = np.dot(next_weights.T,dCdz)*current_layer.ReLUDerivative(z)
                current_layer.computeGradients(dCdz,previousActivations)

        return
    
    def updateParams(self,learning_rate):
        # Adjust all weights and biases at the given learning_rate
        for layer in self.layers:
            layer.updateParams(learning_rate)
    
    # Forward pass the input data -> Backward pass the output of the forward pass -> adjust parameters
    # Repeat for the number of iterations 
    def trainNetwork(self,input_data,expected_values,learn_rate,batch_size,epochs):

        original_batch_size = batch_size

        for i in range(epochs):
            batch_size = original_batch_size

            shuffled_indices = np.random.permutation(input_data.shape[1])
            input_data = input_data[:,shuffled_indices]
            expected_values= expected_values[:,shuffled_indices]

            for j in range(0,input_data.shape[1],batch_size):
                if j + batch_size > input_data.shape[1]:
                    batch_size = input_data.shape[1]

                forward_pass_outputs = self.forwardPass(input_data[:,i:i+batch_size])
                self.backwardPass(forward_pass_outputs,expected_values[:,i:i+batch_size])
                self.updateParams(learn_rate)
    
    # Test a single image (takes the first image of all passed images), draw it and forward pass it throught the network
    def testInput(self,input_data):
        input_data = input_data[:,[0]]

        current_image = input_data.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

        output_data = self.forwardPass(input_data)
        return output_data
    
    def testAccuarcy(self,test_data,expected_values):
        correct_guesses = 0
        for i in range(test_data.shape[1]):
            if np.argmax(self.forwardPass(test_data[:,[i]])) == np.argmax(expected_values[:,i]):
                correct_guesses += 1
        if correct_guesses > 0:
            accuracy = correct_guesses / test_data.shape[1]
            return accuracy
        else:
            return 0