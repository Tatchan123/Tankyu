import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from data.load import load_mnist
from load_cifar10 import load_cifar10
from trainer import *
from model import *
from params_init import *

data_n = 100
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:data_n], "t_test":t_test[:data_n]}

layer = [100,100,100]
opt1 = {"opt":"sgd","lr":0.04,"batch_size":64}
toba_option = { "epsilon":[1e-6,3e-3,1.5e-3,1.2e-3],
                "complement":True,
                "rmw_layer":[2,3,4],
                "delete_n":[0,10,10,10] }
                
network = Convnetwork(input_size=[784], output_size=10, dense_layer=layer, weightinit=He)
test1 = Trainer(network, step=[100,"rmw"], optimizer=opt1, data=data, check=50, tobaoption=toba_option)
result = test1.fit()





