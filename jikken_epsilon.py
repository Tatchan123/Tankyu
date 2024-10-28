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

sgd = Trainer(step=[50], layer=[100,100,100], weightinit=He, optimizer="sgd", data=data, batch_size=100, lr=0.04, check=50)
sgd_result.append(sgd.fit())
parameter = copy.deepcopy(sgd.params)

for i in range(0,10):
    for j in range(0,10):
        for k in range(0,10):
           epsilon = [1,0.00004*(2.5**k),0.00004*(2.5**i),0.0002*(2.5**j)] 
           print(epsilon)
           
           cpparams = copy.deepcopy(parameter)
           complement = Trainer(step=["rmw"], layer=[100,100,100], weightinit=cpparams, optimizer="sgd", data=data, batch_size=100, lr=0.001, 
                                check=50, epsilon=epsilon, complement=True, rmw_layer=[2,3,4])
           result = complement.fit()
           
           tmp = [complement.params["W1"].shape[0]]
           for i in range(1,int(3)):
                tmp = np.append(tmp,complement.params["b"+str(i)].shape)
           complement_result.append([result,tmp])
           print (complement_result)
        
           