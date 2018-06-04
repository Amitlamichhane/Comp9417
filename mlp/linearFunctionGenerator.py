import numpy as np
from random import seed
from random import random
import itertools


#input generator as boolean values 
def matrix_generator(num_input):
    return np.array(list(itertools.product([1,0],repeat=num_input)))


"""
Generate  input and expected ouput
"""
def function_generator(input , Linear_function= False):
    output = []
    arr = matrix_generator(input)
    hidden_layer= 3 
    epoch= 800
    learning_rate= 0.67
    if (not Linear_function):
        
        for i in arr:
            #output.append(i[0] ^ i[1])
            output.append((i[0] and i[1] and i[2] and i[3]) ^ (i[4] and i[5] and i[6] and i[7]) )
        return arr,np.array(output).reshape(len(output),1),hidden_layer,epoch,learning_rate
    else:
        hidden_layer = 0 
        epoch = 200

        for i in arr:
            #output.append(i[0] ^ i[1])
            output.append((i[0] and i[1] and i[2] and i[3]) and (i[4] and i[5] and i[6] and i[7]) )
        return arr ,np.array(output).reshape(len(output),1),hidden_layer , epoch, learning_rate

