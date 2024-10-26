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



origin_model = SGD(layer=[100,100,100], weightinit=He, data_n=5000,
            max_epoch=400, batch_size=500,lr=0.04, check=50)
origin_model.fit()

origin_params = origin_model.params

sake = Toba()

#処理前
print(origin_model.acc())


x_batch = x_train[np.random.choice(origin_model.data_n,origin_model.batch_size)]
#メモリが足りないのでミニバッチで実行

#Tobaのみ
onlytoba_params = sake.rmw(x=x_batch, params=origin_params, epsilon=[1e-6,1e-2,1e-2,1.2e-2],
                                  complement=False, rmw_layer=[2,3,4])
print(origin_model.acc(onlytoba_params))

sake = Toba()

#complement
complement_model = sake.rmw(x=x_batch, params=origin_params, epsilon=[1e-6,1e-2,1e-2,1.2e-2],
                                  complement=True, rmw_layer=[2,3,4])
print(origin_model.acc(complement_model))



#ランダム削除
ran_del_model = sake.rmw_random(params=origin_params,deleat_n=[0,0,10,10,10],rmw_layer=[2,3,4])
print(origin_model.acc(onlytoba_params))
for i in range(1,int((len(ran_del_model)/2)+1)):
    tmp = np.append(tmp,ran_del_model["b"+str(i)].shape)
print(tmp)