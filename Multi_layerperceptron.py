#multilayer perceptron 


import numpy as np
import itertools
import random

def matrix_generator(num_input):
    return np.array(list(itertools.product([1,0],repeat=num_input)))

#boolean function 2
# and(1-4) xor and(5-8)
def bool_f2(input):
    output = []
    for i in input:
        output.append((i[0] and i[1] and i[2] and i[3]) ^ (i[4] and i[5] and i[6] and i[7]) )
    return output

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
	return x*(1.0 - x)

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l=len(self.inputs)
        self.li=len(self.inputs[0])

        self.wi=np.random.random((self.li, self.l))
        self.wh=np.random.random((self.l, 1))

    def think(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))
        return s2

    def train(self, inputs,outputs, it):
        for i in range(it):
            l0=inputs
            l1=sigmoid(np.dot(l0, self.wi))
            l2=sigmoid(np.dot(l1, self.wh))

            l2_err=outputs - l2
            print(l2.shape)
            l2_delta = np.multiply(l2_err, sigmoid_der(l2))
            print(l2_delta.shape)
            print(self.wh.shape)
            l1_err=np.dot(l2_delta, self.wh.T)

            l1_delta=np.multiply(l1_err, sigmoid_der(l1))

            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)




inputs=np.array([[0,0], [0,1], [1,0], [1,1]])
outputs=np.array([ [0], [1],[1],[0] ])
input_matrix = matrix_generator(8)
output_matrix = bool_f2(input_matrix)

"""
n=NN(inputs)
print(n.think(inputs))
n.train(inputs, outputs, 10000)
print(n.think(inputs))
"""
per = NN(input_matrix)
print(per.think(input_matrix))
per.train(input_matrix,output_matrix,100000)
print(per.think(input_matrix))
