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

data_n = 4096
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:data_n], "t_test":t_test[:data_n]}

layer = [100,100,100]
opt1 = {"opt":"adam", "dec1":0.9, "dec2":0.999,"lr":0.002,"batch_size":64}
toba_option = { "rmw_type":"corrcoef",
                "epsilon":[0.6,0.6,0.6,0.6],
                "complement":True,
                "rmw_layer":["Affine2","Affine3","Affine4"],
                "delete_n":[0,40,40,40] }
                
network = Convnetwork(input_size=[784], output_size=10, dense_layer=layer, weightinit=He,drop_rate=[0,0.5])
test1 = Trainer(network, step=[50,"corrcoref_rmw"], optimizer=opt1, data=data, check=50, tobaoption=toba_option)
result = test1.fit()





