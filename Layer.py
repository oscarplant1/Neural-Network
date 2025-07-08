#%%
import numpy as np 
from matplotlib import pyplot as plt

class Layer:
    def __init__(self, nodeInputCount, nodeCount):
        #Number of nodes feeding into the layer - also the number of weights leading into each node
        self.nodeInputCount = nodeInputCount
        #Number of nodes in the layer
        self.nodeCount = nodeCount

        #Initialise weight matrix and bias vector for layer between -0.5 and 0.5
        self.weightsIn = np.random.rand(nodeCount,nodeInputCount) - 0.5
        self.biasesIn = np.random.rand(nodeCount,1) - 0.5

        #Intialise Weight gradients matrix and bias gradients vector to zeros
        self.dCdW = np.zeros([nodeCount,nodeInputCount])
        self.dCdb = np.zeros([nodeCount,1])

        #Empty arrays to keep track of weights and biases
        self.inputs = []
        self.weightedInputs = []
    
    #Using ReLU as hidden layer activation function
    def ReLU(self,z):
        return np.maximum(0,z)

    def ReLUDerivative(self,z):
        return (z > 0).astype(float)

    #Using softMax as final layer activation function, shifting to avoid very large values
    def softMax(self,z):
        shifted = np.exp(z - np.max(z,0))
        A = shifted/np.sum(shifted,0)
        return A

    #Calculate output of a hidden layer
    def calculateOutputs(self,inputs):

        # z = w*a_(L-1) + b
        weightedOutputs = self.weightsIn.dot(inputs) + self.biasesIn
        # a_(L) = ReLU(z)
        activatedOutputs = self.ReLU(weightedOutputs)

        self.inputs.append(inputs)
        self.weightedInputs.append(weightedOutputs)

        return weightedOutputs, activatedOutputs
    
    #Calculate output of a final layer
    def calculateOutputLayer(self,inputs):
        
        # z = w*a_(L-1) + b
        weightedOutputs = self.weightsIn.dot(inputs) + self.biasesIn
        # a_(L) = softMax(z)
        activatedOutputs = self.softMax(weightedOutputs)

        self.inputs.append(inputs)
        self.weightedInputs.append(weightedOutputs)

        return weightedOutputs, activatedOutputs

    def computeGradients(self,dCdZ,previousActivations):
        # dC/dz_(L) = da_(L)/dz_(L) * dC/da_(L)

        # dc/dw_(L) = dz_(L)/dw_(L) * da_(L)/dz_(L) * dC/da_(L) = dz_(L)/dw_(L) * dC/dz_(L)
        #           = a_(L-1) * dC/dz_(L)
        self.dCdW = np.dot(dCdZ,previousActivations.T) / previousActivations.shape[1]

        # dc/db_(L) = dz_(L)/db_(L) * da_(L)/dz_(L) * dC/da_(L) = dz_(L)/db_(L) * dC/dz_(L)
        #           = 1 * dC/dz_(L)
        self.dCdb = np.sum(dCdZ,axis=1,keepdims=True) / previousActivations.shape[1]

    def updateParams(self,learning_rate):
        #Step both the weights and biases of the layer a small amount in the direction of the calculated gradients for the layer
        self.weightsIn = self.weightsIn - learning_rate * self.dCdW
        self.biasesIn = self.biasesIn - learning_rate * self.dCdb 

# %%
