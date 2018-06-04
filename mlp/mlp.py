import numpy as np
from random import seed
from random import random
import itertools
import linearFunctionGenerator as lf
import neurons as n
import os 

def check_values(a ,b):
    print (np.where(predicted_values_check[0]>0.5,0,1) == b)

def clean_files(filename):
    open(filename, 'w').close()
if __name__ =="__main__":
    
    # Test training backprop algorithm
    """
    Linear function check
    """
    seed(1)
    n_inputs = 8
    n_outputs = 2
    inputs, outputs, hidden_layer,epoch,learning_rate  = lf.function_generator(n_inputs,False)

    #joining data 
    data = np.concatenate((inputs, outputs), axis = 1)
    #cleaning files 
    #clean_files("example.txt")
    network = n.initialize_network(n_inputs, hidden_layer, n_outputs)
    n.train(network, data, learning_rate,epoch, n_outputs)
    
    print(n.predict(network,inputs))

    for i in range(len(inputs)):
        predicted_values_check = np.array(n.forward_propagate(network,inputs[i]))
        check_values(predicted_values_check ,outputs[i])
    
