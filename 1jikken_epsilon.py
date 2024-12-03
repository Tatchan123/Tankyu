import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
from data.load import load_mnist
import openpyxl
import copy
from model1 import *
from trainer import *

"""
    self, step, layer, weightinit, optimizer, data, batch_size, lr, check, (decreace1, decreace2, epsilon, complement, rmw_layer, delete_n)
"""
print("start")



data_n = 2000
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:data_n], "t_test":t_test[:data_n]}

layer_set=[[100,100,100],]

layer1 = Trainer(step=[400,"auto_epsilon"], layer=[100,100,100], weightinit=He, optimizer="sgd", data=data, batch_size=200, lr=0.04, check=50, complement=True, rmw_layer=[2,3,4])
layer1.fit()

layer2 = Trainer(step=[400,"auto_epsilon"], layer=[400,200,100], weightinit=He, optimizer="sgd", data=data, batch_size=200, lr=0.04, check=50, complement=True, rmw_layer=[2,3,4])
layer2.fit()

layer3 = Trainer(step=[400,"auto_epsilon"], layer=[150,70,50], weightinit=He, optimizer="sgd", data=data, batch_size=200, lr=0.04, check=50, complement=True, rmw_layer=[2,3,4])
layer3.fit()


