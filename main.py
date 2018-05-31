import arff, numpy as np
import pandas as pd
from scipy.io.arff import loadarff 
import matplotlib.pyplot as plt 



def draw_scatter_matrix(df):
    axes = pd.tools.plotting.scatter_matrix(df, alpha=0.2)
    plt.tight_layout()
    plt.savefig('realationGraph.png')

#read arf files and convert it into numpy 
def read_arf_data (file_name):
    dataset = arff.load(open(file_name, 'r'))
    numpy_data = np.array(dataset['data'])#read like a numpy array 

    raw_data = loadarff(file_name) 
    df_data = pd.DataFrame(raw_data[0]) #create a data_frame with attributes 

    return numpy_data , df_data
    


if __name__ == '__main__':
    data, data_frame= read_arf_data('autoMpg.arff')
    #dealing with null values in the data set     
    #since only few rows are associated with null values we are dropping that rows 
    data_frame = data_frame.dropna(how='any')
    ##for reporting use only 
    draw_scatter_matrix(data_frame)
    
    ##now this steps help in reporting     

