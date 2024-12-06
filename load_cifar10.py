
import pickle as cPickle
from collections import OrderedDict
from gpu import *
if Use_Gpu:
    import cupy as np
else:
    import numpy as np
import os


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict

def change_one_hot_lebel(X):
    T = np.zeros((len(X),10))
    for idx, row in enumerate(T):
        row[X[idx]]=1
    
    return T

def load_cifar10(normalize=False,means=None,stds=None):
    print("cifar-10 dataset loading...")

    fname = os.path.join("./data_cifar10/test_batch")
    test_dict = unpickle(fname)
    x_test = test_dict['data'].reshape(10000,3,32,32)
    if Use_Gpu:
        x_test = np.asarray(x_test)
    t_test = test_dict['labels']
    t_test = change_one_hot_lebel(t_test)
    

    x_train = {}
    t_train = {}

    for idx in range(1,6):
        fname = os.path.join("./data_cifar10/data_batch_"+str(idx))
        train_dict = unpickle(fname)
        # バッチサイズをいじれるように最初は全部くっつけとく
        if idx == 1:
            x_train = train_dict['data'].reshape(10000,3,32,32)
            t = train_dict['labels']
            t_train = change_one_hot_lebel(t)
        else:
            x_train = np.append(x_train, train_dict['data'].reshape(10000,3,32,32),axis=0)
            t = train_dict['labels']
            t_train = np.append(t_train,change_one_hot_lebel(t),axis=0)
    

    if normalize:
        means = np.asarray(means)
        stds = np.asarray(stds)
        x = np.transpose(x_train,(1,0,2,3)).reshape(3,-1)
        mu = np.mean(x,axis=1)
        sigma = np.var(x,axis=1)
        x_train = (x - mu.reshape(3,1)) / np.sqrt(sigma + 1e-9).reshape(3,1) * stds.reshape(3,1) + means.reshape(3,1)
        x_test = (np.transpose(x_test,(1,0,2,3)).reshape(3,-1) - mu.reshape(3,1)) / np.sqrt(sigma + 1e-9).reshape(3,1) * stds.reshape(3,1) + means.reshape(3,1)
        x_train = x_train.reshape(3,50000,32,32).transpose(1,0,2,3)
        x_test = x_test.reshape(3,10000,32,32).transpose(1,0,2,3)

    t_test = t_test.astype(int)
    t_train = t_train.astype(int)
    print("Done")
    return x_train, t_train, x_test, t_test
