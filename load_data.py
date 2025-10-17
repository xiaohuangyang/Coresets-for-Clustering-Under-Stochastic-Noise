from joblib import Memory
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

DATA_PATH = './data/'
mem = Memory('./data/mycache')

 
@mem.cache
def get_data(data_name):
    data = load_svmlight_file(DATA_PATH + data_name + '/' + data_name)
    return data[0], data[1]


def obtain_data(data_name):
    X, y = get_data(data_name)
    X = csr_matrix.toarray(X)
    # X, y = shuffle(X, y, random_state=0)
    return X, y


def obtain_data_new(data_name):
    data = np.loadtxt(DATA_PATH + data_name + '/' + data_name + '.txt')
    return data


def scale_data(X):
    scaler = StandardScaler()
    scaler.fit(X)
    X_scale = scaler.transform(X)
    return X_scale
