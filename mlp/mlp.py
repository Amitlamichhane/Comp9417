import numpy as np
from random import seed
from random import random
import itertools
import linearFunctionGenerator as lf
import nlp as nlp
import neurons as n



def check_values(a ,b):
    print (np.where(predicted_values_check[0]>0.5,0,1) == b)


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
    network = n.initialize_network(n_inputs, hidden_layer, n_outputs)
    
    n.train_network(network, data, learning_rate,epoch, n_outputs)
    """
    for i in range(len(inputs)):
        predicted_values_check = np.array(n.forward_propagate(network,inputs[i]))
        check_values(predicted_values_check ,outputs[i])
    """
