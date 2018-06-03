import gradient_descent as gd
import featureExtraction as fe
import clean_data as cd
#import matplotlib.pyplot as plt 
#import seaborn as sb
import numpy as np



   
    
if __name__ == '__main__':
    #read data from 
    data_frame= cd.read_arf_data('autoMpg.arff')
    #horse power median values is 93.5
    #since we know that the only nan values in the system is horsepower  
    cd.add_median_values(data_frame)
    cd.convert_discontinuous_variable(data_frame)
    data_frame = (data_frame - data_frame.mean())/data_frame.std()
    data_frame.rename(columns={"class": "MPG"},inplace = True)

    data_frame = data_frame.drop(["weight"],axis=1)
    #extract dependent variable from the data 
    y = (data_frame["MPG"].values)
    y= y.reshape(398,1)

    y_column = "MPG"
    
    X = (data_frame.loc[:, data_frame.columns != "MPG"])
    X = (X.iloc[:,0:2]).values
    X_column = (data_frame.loc[:, data_frame.columns != "MPG"])
    x_column = X_column.columns.values
    #add y columns 
    X = gd.add_y_intercept(X)
    theta = np.matrix(np.zeros([X.shape[1],1]))
    
    print(theta.shape)
    print(X.shape)
    print(y.shape)
    a_shape_before = X.shape
    a_shape_after = X[np.logical_not(np.is_nan(X))].shape
    assert a_shape_before == a_shape_after
    #set hyper parameters
    alpha = 0.01
    iters = 100

    g, cost = gd.gradient_descent(X,y,theta,alpha,iters)
   