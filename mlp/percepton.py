import numpy as np
import random
import itertools

#generate input data
def matrix_generator(num_input):
    return np.array(list(itertools.product([1,0],repeat=num_input)))

#boolean function 1
# and(1-4) or and(5-8)
def bool_f1(input):
    output = []
    for i in input:
        #output.append(i[0] ^ i[1])
        output.append((i[0] and i[1] and i[2] and i[3]) or (i[4] and i[5] and i[6] and i[7]) )
    return output



#sigmoid activation function
def sigmoid(x, weights):
    #evaluate inner product
    h = np.dot(x,weights)
    #apply sigmoid activation and return its value
    return 1/(1+np.exp(-h))

#step activation function
#step function with 1 if > 0.5 else 0
def activation_f(x, weights):
    return np.where(np.dot(x,weights)>0.5,1,0)    
    
#split data into training and test sets
def split_data(x,y, split = 70): #split defined as training split i.e split = 70 => 70 train / 30 test
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    for i in range(len(x)):
        if random.randrange(0,100) < split:
            if len(train_x)/len(x) < split/100:
                train_x.append(x[i])
                train_y.append(y[i])
            else:
                test_x.append(x[i])
                test_y.append(y[i])
        else:
            if len(test_x)/len(x) < (100-split)/100:
                test_x.append(x[i])
                test_y.append(y[i])
            else:
                train_x.append(x[i])
                train_y.append(y[i])
                
    return train_x, train_y, test_x, test_y

#train weights with pocket algo
def best_train_weights(x, y, eta, num_epochs): #, x_test, y_test):
    #initialise weights as randoms
    nIn = np.shape(x)[1]    # no. of columns of data matrix
    nOut = np.shape(y)[1]   # no. of columns of class values
    rows = len(x) # no of data points
    #weights = np.random.rand(nIn,nOut)*0.1-0.05
    
    weights = np.zeros((num_inputs,1))
    #add bias weight
    weights = np.append(weights, [[1]], axis=0)
    
    
    best_weights = np.array([i for i in weights])
    best_perror = 1
    for epoch in range(num_epochs):
            #apply activation function
            y_hat = activation_f(x, weights)
            
            #y-y_hat == 0: -> correct classification
            # Update weights for all incorrect classifications
            weights += eta*np.dot(np.transpose(x),y-y_hat)
            # Check current performance
            errors=y-y_hat
            perrors=((rows - np.sum(np.where(errors==0,1,0)))/rows)
            
            #if no errors return the weights
            if perrors == 1:
                return weights, perrors 
            
            
            #output current performance
#             print(perrors, 'is Error on iteration:', epoch)
#             print('Iteration:', epoch, ' Error:', perrors)
#             print('weights', weights)
            
            #find best weights
            if perrors < best_perror:
                best_perror = perrors
                best_weights = np.array([i for i in weights])
    return best_weights, best_perror
    

#simple train weights
def simple_train_weights(x, y, eta, num_epochs): #, x_test, y_test):
    #initialise weights as randoms
    nIn = np.shape(x)[1]    # no. of columns of data matrix
    nOut = np.shape(y)[1]   # no. of columns of class values
    rows = len(x) # no of data points
    #weights = np.random.rand(nIn,nOut)*0.1-0.05
    
    weights = np.zeros((num_inputs,1))
    #add bias weight
    weights = np.append(weights, [[1]], axis=0)
    
    for epoch in range(num_epochs):
            #apply activation function
            y_hat = activation_f(x, weights)
            # Update weights for all incorrect classifications
            weights += eta*np.dot(np.transpose(x),y-y_hat)
            # Check current performance
            errors=y-y_hat
            perrors=((rows - np.sum(np.where(errors==0,1,0)))/rows)
            
    return weights, perrors
    
    
if __name__ == '__main__':
    random.seed(2)
    num_inputs = 8
    input_vals = matrix_generator(num_inputs)
    target = bool_f1(input_vals)
    #add bias term at x_n (last/right-most column)
    rows = len(target)
    x = np.concatenate((input_vals, np.ones((rows,1))),axis=1)
    #reshape target into column matrix
    y = np.array((target)).reshape(rows,1)
    
    #split and train
    eta = 0.01
    n_epochs = 30
    #simple train weights
    train_x, train_y, test_x, test_y = split_data(x,y,70)
    w, e = simple_train_weights(train_x, train_y, eta, n_epochs)
    print()
    print('learning rate: ', eta, 'n_epochs: ', n_epochs)
    print('------------------------------')
    print('Simple Perceptron Training Algorithm')
    print('Weights:')
    print(w)
    print()
    print('perrors: ', e)
    predicted_y = activation_f(test_x, w)
    print('accuracy: ',100*sum(predicted_y == test_y)/len(test_x))
    print()
    print('------------------------------')
    print('Perceptron Training with Pocket Algorithm')
    print('Weights:')

    #train weights with pocket algo
    w2, e2 = best_train_weights(train_x, train_y, eta, n_epochs)
    print(w2)
    print('perrors: ', e2)
    predicted_y2 = activation_f(test_x, w)
    print('accuracy: ',100*sum(predicted_y2 == test_y)/len(test_x))