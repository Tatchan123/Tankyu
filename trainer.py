import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
from data.load import load_mnist
from model1 import *
from weight import *
import matplotlib.pyplot as plt
import tqdm
import copy
import time

(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)


class SGD:
    """
    layer・・・隠れ層 リスト e.g.[100,100,100]
    weightinit・・・重み初期化 クラス e.g.He
    data_n・・・使用するデータの個数 int e.g.1000
    max_epoch・・・何エポック学習回すか int e.g.100
    batch_size・・・バッチサイズ int e.g.100
    lr・・・学習率 float e.g. 0.1
    check・・・何エポックごとに評価するか int e.g. 5          ( batch_size < data_n < batch_size )で頼む
    """
    def __init__(self, layer, weightinit, data_n, max_epoch, batch_size, lr, check):
        
        self.data_n = data_n
        self.x_train = x_train[:self.data_n]
        self.x_test = x_test[:self.data_n]
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.check = check
        
        wi = weightinit()
        self.params = wi.weight_initialization(inp=784, layer=layer, out=10)
        
        self.model = Network(input_size=784, output_size=10, layer_size=layer, params=self.params, activation=Relu)
        
        self.train_acc = []
        
    def fit(self):
        cnt = 0
        for i in range(self.max_epoch):
            for j in range(self.data_n // self.batch_size):
                batch_mask = np.random.permutation(np.arange(self.data_n))
                x_batch = x_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)-1]]
                t_batch = t_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)-1]]
                # ↑ここ、前のだとバッチ1つ作って1エポックってことになってたから全データ使うようにした
                
                grads = self.model.gradient(x_batch, t_batch, self.params)      #gradientメソッド呼び出された時点でNetwork側のparamsを初期化 対応できないところ結構ありそう（かなりめんどい）
                for key in self.params.keys():
                    self.params[key] -= self.lr*grads[key]
            
            if cnt == self.check:
                cnt = 0
                tmp = self.model.accuracy(x_train,t_train)
                self.train_acc.append(tmp)
                print("epoch:",str(i)," | ",str(tmp))
            cnt += 1

    def acc(self,params=None):
        if params is not None:
            self.params=params
        self.model.updateparams(self.params)
        return self.model.accuracy(x_train,t_train)


class CpSGD:
    """
    epsilon,complement: Network.rmw参照
    cp: 特徴量比較して結合する作業をやる回数
    """
    def __init__(self, layer, weightinit, data_n, epochs, batch_size, lr, check, epsilon, complement, rmw_layer):
        
        self.layer = layer
        self.data_n = data_n
        self.x_train = x_train[:self.data_n]
        self.x_test = x_test[:self.data_n]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.check = check
        self.epsilon = epsilon
        self.complement = complement
        self.rmw_layer = rmw_layer

        wi = weightinit()
        self.params = wi.weight_initialization(inp=784, layer=layer, out=10)
        #self.dictshape(self.params)
        
        self.model = Network(input_size=784, output_size=10, layer_size=layer, params=self.params, activation=Relu, toba=True)
        
        self.train_acc = []
        
    def fit(self):
        cnt = 0
        for single_epoch in self.epochs:
            for i in range(single_epoch):
                for j in range(self.data_n // self.batch_size):
                    batch_mask = np.random.permutation(np.arange(self.data_n))
                    x_batch = x_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)-1]]
                    t_batch = t_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)- 1]]

                    grads = self.model.gradient(x_batch, t_batch, self.params)      
                for key in self.params.keys():
                    self.params[key] -= self.lr*grads[key]
                
                if cnt == self.check:
                    cnt = 0
                    print("epoch:",str(i)," | ",str(self.model.accuracy(x_train,t_train)))
                cnt += 1
                
                
            self.model.updateparams(self.params)
            print("start rmw from:"+str(self.model.accuracy(x_train,t_train))," ===========================================")
            self.params = self.model.layers["toba"].rmw(x=x_batch, params=self.params, epsilon=self.epsilon, complement=self.complement, rmw_layer=self.rmw_layer)
            self.model.updateparams(self.params)
            #↓表示用
            tmp = [self.params["W1"].shape[0]]
            for i in range(1,int((len(self.params)/2)+1)):
                tmp = np.append(tmp,self.params["b"+str(i)].shape)
            print(tmp)
            print("after rmw:",str(self.model.accuracy(x_train,t_train)))
            print("finish rmw ==========================================")
        print("finish")
    def acc(self,params=None):
        if params is not None:
            self.params=params
        self.model.updateparams(self.params)
        return self.model.accuracy(x_train,t_train)
class Adam:
    def __init__(self, layer, weightinit, data_n, max_epoch, batch_size, lr, check, decreace1, decreace2):
        
        self.data_n = data_n
        self.x_train = x_train[:self.data_n]
        self.x_test = x_test[:self.data_n]
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.check = check
        self.decreace1 = decreace1
        self.decreace2 = decreace2
        
        wi = weightinit()
        self.params = wi.weight_initialization(inp=784, layer=layer, out=10)
        
        self.model = Network(input_size=784, output_size=10, layer_size=layer, params=self.params, activation=Relu)
        
        self.train_acc = []
        
    def fit(self):
        cnt = 0
        self.m,self.v = {},{}
        for k,v in self.params.items():
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)
            
        for i in range(self.max_epoch):
            for j in range(self.data_n // self.batch_size):
                batch_mask = np.random.permutation(np.arange(self.data_n))
                x_batch = x_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)-1]]
                t_batch = t_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)-1]]
                grads = self.model.gradient(x_batch, t_batch, self.params)

            crt_lr = self.lr * np.sqrt(1.0 - self.decreace2**(cnt+1)) / (1.0 - self.decreace1**(cnt+1))
            
            for key in self.params.keys():
                self.m[key] += (1-self.decreace1) * (grads[key]-self.m[key])
                self.v[key] += (1-self.decreace2) * (grads[key]**2-self.v[key])
                self.params[key] -= crt_lr * self.m[key] / (np.sqrt(self.v[key]) +1e-7)
                
            if cnt == self.check:
                cnt = 0
                tmp = self.model.accuracy(x_train,t_train)
                self.train_acc.append(tmp)
                print("epoch:",str(i)," | ",str(tmp))
            cnt += 1


class CpAdam:
    def __init__(self, layer, weightinit, data_n, epochs, batch_size, lr, check, decreace1, decreace2, epsilon, complement, rmw_layer):
        
        self.layer = layer
        self.data_n = data_n
        self.x_train = x_train[:self.data_n]
        self.x_test = x_test[:self.data_n]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.check = check
        self.decreace1 = decreace1
        self.decreace2 = decreace2
        self.epsilon = epsilon
        self.complement = complement
        self.rmw_layer = rmw_layer
        
        wi = weightinit()
        self.params = wi.weight_initialization(inp=784, layer=layer, out=10)
        
        self.model = Network(input_size=784, output_size=10, layer_size=layer, params=self.params, activation=Relu)
        
        self.train_acc = []
        
    def fit(self):
        cnt = 0
        self.m,self.v = {},{}

        for k,v in self.params.items():
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)


        for single_epoch in self.epochs:        
            for i in range(single_epoch):
                for j in range(self.data_n // self.batch_size):
                    batch_mask = np.random.permutation(np.arange(self.data_n))
                    x_batch = x_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)-1]]
                    t_batch = t_train[batch_mask[self.batch_size*j : self.batch_size*(j+1)-1]]
                

                    grads = self.model.gradient(x_batch, t_batch, self.params)

                crt_lr = self.lr * np.sqrt(1.0 - self.decreace2**(cnt+1)) / (1.0 - self.decreace1**(cnt+1))
                
                for key in self.params.keys():
                    self.m[key] += (1-self.decreace1) * (grads[key]-self.m[key])
                    self.v[key] += (1-self.decreace2) * (grads[key]**2-self.v[key])
                    self.params[key] -= crt_lr * self.m[key] / (np.sqrt(self.v[key]) +1e-7)
                    
                if cnt == self.check:
                    cnt = 0
                    tmp = self.model.accuracy(x_train,t_train)
                    self.train_acc.append(tmp)
                    print("epoch:",str(i)," | ",str(tmp))
            cnt += 1
            self.params = self.model.layers["toba"].rmw(x=x_batch, params=self.params, layer=self.layer, epsilon=self.epsilon, complement=self.complement, rmw_layer=self.rmw_layer)
            # self.dictshape(self.params)

    def dictshape(sekf,dict):
        for key ,value in dict.items():
            print(key,":",value.shape)


# Wの動きを観察してみよう


# def get_loss(param_key,base_neuron,forward_neuron,w_changes,function):
#     new_params = copy.deepcopy(params)
#     loss_changes = np.array([])
#     for i in w_changes:
#       new_params["W"+str(param_key)][base_neuron][forward_neuron] = params["W"+str(param_key)][base_neuron][forward_neuron].copy() + i
#       optimizer1.model.updateparams(new_params)
#       loss_changes = np.append(loss_changes,eval("trial1.model."+function+"(x_train,t_train)"))
#     #   print(params["W"+str(param_key)][base_neuron][forward_neuron])
#     #   print(new_params["W"+str(param_key)][base_neuron][forward_neuron])
#     return(loss_changes)

# def plt_save(name,base,forward,function,xx):
#     fig = plt.figure()
#     w_changes = xx
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax3 = fig.add_subplot(2, 2, 3)
#     ax4 = fig.add_subplot(2, 2, 4)

#     loss1 = get_loss(1,base,forward,w_changes,function)
#     loss2 = get_loss(2,base,forward,w_changes,function)
#     loss3 = get_loss(3,base,forward,w_changes,function)
#     loss4 = get_loss(4,base,forward,w_changes,function)
    
#     if gpu.Use_Gpu:
#         w_changes = np.ndarray.get(w_changes)
#         loss1,loss2,loss3,loss4 = np.ndarray.get(loss1),np.ndarray.get(loss2),np.ndarray.get(loss3),np.ndarray.get(loss4)

#     ax1.plot(w_changes,loss1)
#     ax1.set_title("1")


#     ax2.plot(w_changes,loss2)
#     ax2.set_title("2")


#     ax3.plot(w_changes,loss3)
#     ax3.set_title("3")


#     ax4.plot(w_changes,loss4)
#     ax4.set_title("4")
    
#     plt.savefig("image/copy/"+name+".png")   












# params = copy.deepcopy(optimizer1.params)


# # for i in tqdm.tqdm(range(0,2)):
# #     # x = np.arange(-1,1,0.05)
# #     # plt_save(str(i)+"accuracy",i,i,"accuracy",x)
# #     x = np.arange(-1,1,0.05)
# #     plt_save(str(i)+"loss",i,i,"cal_loss",x)


# print("finish")