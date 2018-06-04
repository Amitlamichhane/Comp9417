import gradient_descent as gd

import clean_data as cd
import matplotlib.pyplot as plt 
#import seaborn as sb
import numpy as np
import split_data as sp 
from sklearn import model_selection 
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression




   
    
if __name__ == '__main__':
    #read data from 
    data_frame= cd.read_arf_data('autoMpg.arff')
    #horse power median values is 93.5
    #since we know that the only nan values in the system is horsepower  
    data_frame = cd.add_median_values(data_frame)
    #print((data_frame.iloc[:,0:-7].head()))
    cd.convert_discontinuous_variable(data_frame)
    #print((data_frame.iloc[:,21:]).head())
    #normalize otherwise it will overflow 
    data_frame = (data_frame - data_frame.mean())/data_frame.std()
    data_frame.rename(columns={"class": "MPG"},inplace = True)

    data_frame = data_frame.drop(["weight"],axis=1)
    #extract dependent variable from the data 
    y = (data_frame["MPG"].values)
    y= y.reshape(398,1)
    y_column = "MPG"
    X = (data_frame.loc[:, data_frame.columns != "MPG"])
    #X = (X.iloc[:,0:2]).values
    X= X.values

    train_x, train_y, test_x, test_y = sp.split_data(X,y)

    X_column = (data_frame.loc[:, data_frame.columns != "MPG"])
    x_column = X_column.columns.values
    #add y columns 
    X = gd.add_y_intercept(X)
    theta = np.matrix(np.zeros([train_x.shape[1],1]))
    
    
    #set hyper parameters
    alpha = 0.001
    iters = 10000
    
    
    g, cost = gd.gradient_descent(train_x,train_y,theta,alpha,iters)
    
    y_predict = np.dot(test_x,g)
    #print(y_predict)
    mse= np.mean(y_predict- test_y)**2
    rmse = np.sqrt(mse)
    print("MSE using gradient descent: ", mse)
    print("RMSE using gradient descent: ", rmse)
    

    from sklearn.model_selection import train_test_split

    #x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 0)

    model = LinearRegression()
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    #print(y_pred)
    mse_sklearn = np.mean(y_pred - test_y) ** 2
    rmse_sklearn = np.sqrt(mse_sklearn)
    print("MSE using sklearn: ", mse_sklearn)
    print("RMSE using sklearn: ", rmse_sklearn)
    
    ##accuracy test 

    finalCost = gd.cost(train_x,train_y,g)
    print("Accuracy of the linear regression with gradient descent:",(100.0-(finalCost*100)))
    
    
 
    
