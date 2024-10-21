"""
complementが働いてるかを確かめる実験
ほかにもいろいろ実験したいから一応ファイルわけとく
"""

import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict

import copy
from model1 import *
from trainer import *

nomal_model = SGD(layer=[100,100,100], weightinit=He, data_n=5000,
            epochs=400, batch_size=500,lr=0.04, check=50)
nomal_model.fit()

#処理前
print(nomal_model.acc())
#Tobaのみ
onlytoba_model = copy.deepcopy(nomal_model)
print(onlytoba_model.acc(Toba.rmw(x=x_train, params=onlytoba_model.params, epsilon=[1e-6,1e-2,1e-2,1.2e-2],
                                  complement=False, rmw_layer=[2,3,4])))
#complement
complement_model = copy.deepcopy(nomal_model)
print(complement_model.acc(Toba.rmw(x=x_train, params=onlytoba_model.params, epsilon=[1e-6,1e-2,1e-2,1.2e-2],
                                    complement=False, rmw_layer=[2,3,4])))
#ランダム削除
ran_del_model = copy.deepcopy(nomal_model)
print()