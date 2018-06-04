import random
import numpy as np
#split data into training and test sets
def split_data(x,y, split = 70): #split defined as training split i.e split = 70 => 70 train / 30 test
    random.seed(0)
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
                
    return np.array(train_x), np.array(train_y), np.array(test_x), np.array(test_y)