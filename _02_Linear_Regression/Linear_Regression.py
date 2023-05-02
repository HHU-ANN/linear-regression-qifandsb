# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    # X, y = read_data()
    # weight = np.linalg.inv(np.matmul(X.T,X)) * np.matmul(X.T, y)
    # return weight @ data
    x, y = read_data()
    i = np.eye(6)
    h = 10000000
    print(h)
    w = np.dot(np.linalg.inv(np.dot(x.T, x) + h * i), np.dot(x.T, y))
    m = np.dot(x.T, y) - np.dot(x.T, x)
    print(m)
    return w @ data

def lasso(data):
    x, y = read_data()
    w = np.dot(np.linalg.inv(np.dot(x.T, x)), np.dot(x.T, y))
    return w @ data

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y