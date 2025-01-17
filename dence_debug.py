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


data_n = 8192
(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:1024], "t_test":t_test[:1024]}

layer = [512,256,128]
opt1 = {"opt":"adam", "dec1":0.9, "dec2":0.999,"lr":0.002,"batch_size":64}
toba_option = { "rmw_type":"coco_toba",
                "epsilon":[0.0,0.0,0.0,0.0],
                "complement":True,
                "rmw_layer":["Affine1","Affine2","Affine3","Affine4"],
                "delete_n":[196,128,64,32] ,
                "strict":False}
                
network = Convnetwork(input_size=[784], output_size=10, dense_layer=layer, weightinit=He, toba=True, drop_rate=[0,0.5], regularize=["l2",0.0005])
test1 = Trainer(network, step=[20,"corrcoref_rmw",10], optimizer=opt1, data=data, check=10, tobaoption=toba_option)
result = test1.fit()



