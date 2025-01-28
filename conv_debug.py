import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from data.load import load_mnist
from load_cifar10 import load_cifar10

from model import *
from trainer import *
from params_init import *

data_n = 8192


#x_train, t_train, x_test, t_test = load_cifar10(normalize=True, means=[0.5,0.5,0.5], stds=[0.5,0.5,0.5])
(x_train, t_train),(x_test, t_test) = load_mnist(normalize=True,flatten=False)
data = {"x_train":x_train[:data_n], "t_train":t_train[:data_n], "x_test":x_test[:256], "t_test":t_test[:256]}


layer1 = [512,256,128]
conv_layer1 = [[32,3,1],[2],[64,3,1],[2],[128,3,1],[2]]
opt2 = {"opt":"adam", "dec1":0.9, "dec2":0.999,"lr":0.002,"batch_size":64}
exp = {"method":"exp", "base":0.92}

network = Convnetwork(input_size=(list(x_train[0].shape)), output_size=10, dense_layer=layer1, conv_layer=conv_layer1, weightinit=He, activation=Relu, batchnorm=True, toba=True, drop_rate=[0.26,0.33], regularize=["l2",0.0005])
test = Trainer(network, optimizer=opt2, data=data, check=5, scheduler=exp)
test.fit(10)
#prev = copy.deepcopy(test)

copy.deepcopy(test).rmw_fit("random_rmw",["Affine1","Affine2","Affine3","Affine4"],[256,64,32,16])
test.coco_sort(["Affine1","Affine2","Affine3","Affine4"])
test.rmw_fit("coco_toba",["Affine1","Affine2","Affine3","Affine4"],[256,64,32,16])

# test.rmw_fit("coco_toba",["Affine1","Affine2","Affine3","Affine4"],[10,10,10,10])
#prev.prev_coco_sort(["Affine2","Affine3","Affine4"])
#prev.rmw_fit("coco_toba",["Affine2","Affine3","Affine4"],[400,100,50,25])


