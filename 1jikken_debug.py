import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
from data.load import load_mnist
import copy
from model1 import *
from trainer import *

data_n = 100
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)

data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:data_n], "t_test":t_test[:data_n]}
layer = [100,100,100]
opt1 = {"opt":"sgd"}
toba_option = { "epsilon":[1e-6,3e-3,1.5e-3,1.2e-3],
                "complement":True,
                "rmw_layer":[2,3,4],
                "delete_n":[0,10,10,7] }
                

test1 = Trainer(step=[100,"rmw"], layer=layer, weightinit=He, optimizer=opt1, data=data, batch_size=64, lr=0.04, 
   check=50, tobaoption=toba_option)
test1.fit()
test2 = Trainer(step=[100,"random_rmw"], layer=layer, weightinit=He, optimizer=opt1, data=data, batch_size=64, lr=0.04, 
   check=50, tobaoption=toba_option)
test2.fit()


test3 = Trainer(step=[100,"count_rmw"], layer=layer, weightinit=He, optimizer=opt1, data=data, batch_size=64, lr=0.04, 
   check=50, tobaoption=toba_option)
test3.fit()

test4 = Trainer(step=[100,"auto_epsilon_rmw"], layer=layer, weightinit=He, optimizer=opt1, data=data, batch_size=64, lr=0.04, 
   check=50, tobaoption=toba_option)
test4.fit()




