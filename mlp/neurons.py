import numpy as np
from random import seed
from random import random
import itertools

# sigmoid neuron activation
def sigmoid(activation):
    return 1.0 / (1.0 + np.exp(-activation))

def sigmoid_derivative(output):
    return output * (1.0 - output)



# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    '''n_inputs = number of inputs
       n_hidden = number of hidden layers
       n_outputs = number of outputs
    '''
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]}
                     for i in range(n_hidden)]
    
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} 
                    for i in range(n_outputs)]
    network.append(output_layer)
    return network

def neuron_output(weights, inputs):
    # Calculate output of the linear model
    # sum(Weights*input) + bias i.e Y = w.T*X + bias
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron_output(neuron['weights'], inputs)
            neuron['output'] = sigmoid(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * sigmoid_derivative(neuron['output'])

# Train a network for a fixed number of epochs
def train(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)

            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1

            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            ##update weights
            for i in range(len(network)):
                inputs = row[:-1]
                if i != 0:
                    inputs = [neuron['output'] for neuron in network[i - 1]]
                for neuron in network[i]:
                    for j in range(len(inputs)):
                        neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                    neuron['weights'][-1] += l_rate * neuron['delta']
                         

        #writing data into files                    
        #with open("example.txt","a") as myFile:
        #   myFile.write("%.2f,%.2f \n"%(epoch,sum_error))
       
#prediction formula 
def predict(network, input):
    return(forward_propagate(network, input))