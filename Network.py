import numpy as np 
from matplotlib import pyplot as plt
import Layer

np.set_printoptions(suppress=True)

# Neural Netowrk class using ReLU as the hidden layer activation function, softmax as the final layer activation function and categorical cross entropy loss as the cost/loss function

class Network:
    def __init__(self, nodesPerLayer):
        self.nodesPerLayer = nodesPerLayer

        # Initialise list of layers
        self.layers = []
        for i in range(len(self.nodesPerLayer)-1):
            self.layers.append(Layer.Layer(nodesPerLayer[i],nodesPerLayer[i+1]))
    
    # Using cross-entropy loss for the cost function of the network - not used in current implementation
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
        # Adjust all weights and biases at the given learning rate
        for layer in self.layers:
            layer.updateParams(learning_rate)
    
    # Shuffle data -> divide data into batches of input size -> (forward pass -> backward pass -> update parameters) for each batch -> Repeat for the input number of epochs 
    def trainNetwork(self,training_data,training_expected_values,learn_rate,batch_size,epochs,plot_accuracy = False,test_data = 0,test_expected_values = 0):

        original_batch_size = batch_size

        training_accuracies = []
        test_accuracies = []

        for i in range(epochs):
            batch_size = original_batch_size

            shuffled_indices = np.random.permutation(training_data.shape[1])
            training_data = training_data[:,shuffled_indices]
            training_expected_values= training_expected_values[:,shuffled_indices]

            # Calculate and plot live accuracy for both training and test data
            if plot_accuracy:
                current_training_predictions = self.forwardPass(training_data)
                accuracy = self.computeAccuracy(current_training_predictions,training_expected_values)
                training_accuracies.append(accuracy*100)
                current_test_predictions = self.forwardPass(test_data)
                accuracy = self.computeAccuracy(current_test_predictions,test_expected_values)
                test_accuracies.append(accuracy*100)

                plt.cla()
                plt.plot(range(i+1),training_accuracies,"r-",label= "Training data")
                plt.plot(range(i+1),test_accuracies,"b-", label="Test data")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy %")
                plt.legend(loc="lower right")
                plt.axis([0,epochs-1,0,100])
                plt.xticks(np.arange(0,epochs+1,5))
                plt.yticks(np.arange(0,101,5))
                plt.grid()
                plt.pause(0.05)

            # Training network
            for j in range(0,training_data.shape[1],batch_size):
                # Last batch may be bigger than the rest in order to cover all of the dataset
                if j + batch_size > training_data.shape[1]:
                    batch_size = training_data.shape[1] - j 

                forward_pass_outputs = self.forwardPass(training_data[:,j:j+batch_size])
                self.backwardPass(forward_pass_outputs,training_expected_values[:,j:j+batch_size])
                self.updateParams(learn_rate)

            # Decay the learning rate with every epoch
            learn_rate = learn_rate - 0.001

        # Plot the accuracy after the final epoch
        if plot_accuracy:
            final_training_predictions = self.forwardPass(training_data)
            final_training_accuracy = self.computeAccuracy(final_training_predictions, training_expected_values) * 100
            training_accuracies.append(final_training_accuracy)
            final_test_predictions = self.forwardPass(test_data)
            final_test_accuracy = self.computeAccuracy(final_test_predictions, test_expected_values) * 100
            test_accuracies.append(final_test_accuracy)

            plt.cla()
            plt.plot(range(epochs+1), training_accuracies, "r-", label="Training data")
            plt.plot(range(epochs+1), test_accuracies, "b-", label="Test data")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy %")
            plt.legend(loc="lower right")
            plt.axis([0,epochs,0,100])
            plt.xticks(np.arange(0,epochs+1,5))
            plt.yticks(np.arange(0,101,5))
            plt.grid()
            plt.show()

            print(training_accuracies[-1],test_accuracies[-1])

    # Test a single image (takes the first image of all input images), draw it and forward pass it throught the network
    def testInput(self,input_data):
        input_data = input_data[:,[0]]

        current_image = input_data.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

        output_data = self.forwardPass(input_data)
        return output_data
    
    # Compare the predictions made by the network to the correct values and return the accuracy
    def computeAccuracy(self, predictions, expected_values):
        predicted = np.argmax(predictions, axis=0)
        correct = np.argmax(expected_values, axis=0)
        accuracy = np.mean(predicted == correct)
        return accuracy
    
