
import arff, numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
#import matplotlib.pyplot as plt 
#import seaborn as sb

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

    df1.drop(['origin','model','cylinders'],axis=1,inplace=True)    
    
    return df1
    

def draw_scatter_matrix(df):

    attributes =df.columns.values
    
    g = sb.PairGrid(df, x_vars=attributes[0], y_vars='class')    
    g = g.map(plt.scatter)
    plt.show()
    

#read arf files and convert it into numpy 
def read_arf_data (file_name):
    raw_data = loadarff(file_name) 
    df_data = pd.DataFrame(raw_data[0]) #create a data_frame with attributes 

    return  df_data


#Add median values for missing data 
def add_median_values(df):
    df = df.fillna(df.median())
    

