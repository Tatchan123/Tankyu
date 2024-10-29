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



data_n = 1000
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:data_n], "t_test":t_test[:data_n]}

sgd_result = []
complement_result = []
for i in range(10):

    toba = Trainer(step=[300,"rmw"], layer=[100,100,100], weightinit=He, optimizer="sgd", data=data, batch_size=100, lr=0.04, check=50,
                    epsilon=[1e-6,1e-2,1e-2,1.2e-2], complement=True, rmw_layer=[2,3,4])
    complement_result.append(toba.fit())

print(sgd_result)
print(complement_result)