import arff, numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
import matplotlib.pyplot as plt 
import gradient_descent
import seaborn as sb

#convert discontinuous variable to boolean table
def convert_discontinuous_variable(df):
    cylinder_var=[3,4,5,6,8]
    model_var=[70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82]
    origin_var=[1, 3, 2]
    df1= df
    for k in model_var:
        df1['model'+'_'+str(k)]= df1['model'].str.decode("utf-8")
        df1['model'+'_'+str(k)]= np.where(df1['model'+'_'+str(k)]== str(k),1,0)


    for k in origin_var:
        df1['origin'+'_'+str(k)]= df1['origin'].str.decode("utf-8")
        df1['origin'+'_'+str(k)]= np.where(df1['origin'+'_'+str(k)]==str(k) ,1,0)
    for k in cylinder_var:
        df1['cylinders'+'_'+str(k)]= df1['cylinders'].str.decode("utf-8")        
        df1['cylinders'+'_'+str(k)]= np.where(df1['cylinders'+'_'+str(k)]==str(k) ,1,0)

    df1.drop(['horsepower','model','cylinders'],axis=1,inplace=True)    
    
    return df1
    

def draw_scatter_matrix(df):
    df.plot(kind='line')
    pass

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

    new_data_frame = convert_discontinuous_variable(data_frame)
    #print(data_frame.head())
    #print(new_data_frame.head())
    data_frame.corr()
    sb.heatmap(data_frame.corr())
    plt.show()
    numpy_data = new_data_frame.as_matrix(columns= None) #storing number as a numpy matrix 
    #print(numpy_data)
    attributes =new_data_frame.columns.values #list of column names 
    #print(attributes)

    #draw_scatter_matrix(data_frame)
    #after seeing the scatter plot we can make linear regression architecture for multiple 
    # attributes, let's start with simple attributes and plot the resulting data 
    # gradient_descent with linear regression 
    # multivariable linear regression is our data
