import arff, numpy as np


#read arf files and convert it into numpy 
def read_arf_data (file_name):
    dataset = arff.load(open(file_name, 'r'))
    numpy_data = np.array(dataset['data'])#read like a numpy array 
    return numpy_data
    


if __name__ == '__main__':
    data = read_arf_data('autoMpg.arff')
    print(data)