import arff, numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
import matplotlib.pyplot as plt 
import gradient_descent


def draw_scatter_matrix(df):
    axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
    plt.tight_layout()
    plt.savefig('realationGraph.png')

#read arf files and convert it into numpy 
def read_arf_data (file_name):
    raw_data = loadarff(file_name) 
    df_data = pd.DataFrame(raw_data[0]) #create a data_frame with attributes 

    return  df_data
    


if __name__ == '__main__':
    data_frame= read_arf_data('autoMpg.arff')
    #dealing with null values in the data set     
    #since only few rows are associated with null values we are dropping that rows 

    #•	Replace the missing values of horsepower column with the median value of the same column
    
    #horse power median values is 93.5
    #since we know that the only nan values in the system is horsepower 
    
    data_frame = data_frame.fillna(data_frame.median())

    numpy_data = data_frame.as_matrix(columns= None) #storing number as a numpy matrix 
    attributes =data_frame.columns.values #list of column names 
    draw_scatter_matrix(data_frame)

    
    #after seeing the scatter plot we can make linear regression architecture for multiple 
    # attributes, let's start with simple attributes and plot the resulting data 
    # gradient_descent with linear regression 
    # multivariable linear regression is our data
   

