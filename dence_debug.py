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
import copy


data_n = 8192
data_n = 256
test_n = 256
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:test_n], "t_test":t_test[:test_n]}

layer = [50,50]
opt1 = {"opt":"adam", "dec1":0.9, "dec2":0.999,"lr":0.002,"batch_size":64}
                
network = Convnetwork(input_size=[784], output_size=10, dense_layer=layer, weightinit=He, toba=True, drop_rate=[0,0.5], regularize=["l2",0.0005])
test1 = Trainer(network, optimizer=opt1, data=data, check=10)
print(test1.fit(10))
#test2 = copy.deepcopy(test1)
print(test1.rmw_fit("random_rmw",["Affine1","Affine2","Affine3"],[15,15,15],[0.0,0.0,0.0]))
print(test1.fit(50))
#test1.coco_sort(["Affine1","Affine2","Affine3"])



