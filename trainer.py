import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from model1 import *
from weight import *
import time
import sys


class Trainer:
    def __init__(self, step, layer, weightinit, optimizer, data, batch_size, lr, check, epsilon=None, complement=None, rmw_layer=None, delete_n=None, rmw_n=None):
        
        self.step = step
        self.layer = layer
        self.x_train = data["x_train"]
        self.t_train = data["t_train"]
        self.x_test = data["x_test"]
        self.t_test = data["t_test"]
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.check = check
        self.epsilon = epsilon
        self.complement = complement
        self.rmw_layer = rmw_layer
        self.delete_n = delete_n
        self.rmw_n = rmw_n
        self.optimizer = Optimizer(self.model,optimizer)
        
        try:
            wi = weightinit()
            self.params = wi.weight_initialization(inp=784, layer=layer, out=10)
        except:
            self.params = weightinit
        self.params["init_remove"] = np.array([])
        self.model = Network(input_size=784, output_size=10, layer_size=layer, params=self.params, activation=Relu, toba=True)
        
    def fit(self):
        for step in self.step:
            if type(step) == int:
                self.optimizer.fit(self.data,self.batch_size,step,self.lr,self.check)
            else:
                if step == "rmw":
                    self.params, dacc = self.rmw()
                if step == "random_rmw":
                    self.params = self.random_rmw()
                if step == "count_rmw":
                    self.params = self.count_rmw()
                if step == "auto_epsilon":
                    self.params = self.auto_epsilon_rmw()
        
        self.model.updateparams(self.params)
        t1 = time.time()
        acc = self.model.accuracy(self.x_test,self.t_test)
        t2 = time.time()
        elapsed_time = t2-t1
        return float(dacc)


    
    
    def rmw(self):
        params = self.params
        print("start rmw ===========================================")
        self.model.updateparams(params)
        acc1 = self.model.accuracy(self.x_test,self.t_test)
        print("accuracy before rmw :",str(acc1))
        
        params = self.model.rmw(self.x_train,self.epsilon,self.complement,self.rmw_layer)
        
        self.model.updateparams(params)
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.layer)+2)):
            tmp = np.append(tmp,params["b"+str(i)].shape)
        print("Composition of Network :",tmp)
        acc2 = self.model.accuracy(self.x_test,self.t_test)
        print("accuracy after rmw :",str(acc2))
        print("finish rmw ------------------------------------------")
        return params, acc2-acc1
    
    def count_rmw(self):
        params = self.params
        print("start +++++ COUNT +++++ rmw ===========================================")
        self.model.updateparams(params)
        print("accuracy before rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        
        params = self.model.count_rmw(self.x_train,self.epsilon,self.complement,self.rmw_layer, rmw_n=self.rmw_n)
        
        self.model.updateparams(params)
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.layer)+2)):
            tmp = np.append(tmp,params["b"+str(i)].shape)
        print("Composition of Network :",tmp)
        print("accuracy after rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        print("finish +++++ COUNT +++++ rmw ------------------------------------------")
        return params
    
    
    def random_rmw(self):
        params = self.params
        print("start ????? RANDOM ????? rmw ==========================================")
        print("accuracy before rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        params = self.model.random_rmw(self.x_train, self.rmw_layer, delete_n=self.delete_n)
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.layer)+2)):
            tmp = np.append(tmp,params["b"+str(i)].shape)
        print("Composition of Network :",tmp)
        print("accuracy after rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        print("finish ????? RANDOM ????? rmw -----------------------------------------")
        return params
    
    def auto_epsilon_rmw(self):
        params = self.params
        print("start <<<<< AUTO_EPSILON >>>>> rmw ====================================")
        print("accuracy before rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        params = self.model.auto_epsilon_rmw(self.x_train, self.complement, self.rmw_layer)
        self.model.updateparams(params)
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.layer)+2)):
            tmp = np.append(tmp,params["b"+str(i)].shape)
        print("Composition of Network :",tmp)
        print("accuracy after rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        print("finish <<<<< AUTO EPSILON >>>>> rmw -----------------------------------")
        return params
    

    def measure(self):
        self.model.updateparams(self.params)
        t1 = time.time()
        acc = self.model.predict(self.x_test,self.t_test)
        t2 = time.time()
        elapsed_time = t2-t1

        return elapsed_time

    def get_memory(self): #変数のメモリ使用量を取得(画像1枚、重み、バイアス),単位はバイト
        mmlist = []
        x_single = self.x_train[0]
        mmlist.append(x_single.nbytes)
        
        for p in self.params.values():
            mmlist.append(p.nbytes)
        
        return sum(mmlist)


class Optimizer():
    def __init__(self,model,opt_dict,scheduler_dict=None):
        self.model = model
        opt = opt_dict["opt"]
        if opt == "sgd":
            self.fit = self.SGD
        elif opt == "adam":
            self.fit = self.Adam
            self.decreace1 = opt_dict["dec1"]
            self.decreace2 = opt_dict["dec2"]

    def SGD(self,data,batchsize,maxepoch,lr,check):
        x_train = data["x_train"]
        t_train = data["t_train"]
        x_test = data["x_test"]
        t_test = data["t_test"]
        for i in range(maxepoch):
            batch_mask = np.random.permutation(np.arange(len(x_train)))
            for j in range(len(x_train) // batchsize):
                x_batch = x_train[batch_mask[batchsize*j : batchsize*(j+1)]]
                t_batch = t_train[batch_mask[batchsize*j : batchsize*(j+1)]]
                grads = self.model.gradient(x_batch, t_batch, self.model.params)
                for key in grads.keys():
                    self.model.params[key] -= lr*grads[key]
                
            if (i+1) % check == 0:
                acc = self.model.accuracy(x_test,t_test)
                print("epoch:",str(i)," | ",str(acc))

    def Adam(self,data,batchsize,maxepoch,lr,check):
        x_train = data["x_train"]
        t_train = data["t_train"]
        x_test = data["x_test"]
        t_test = data["t_test"]
        self.m,self.v = {},{}
        for k,v in self.model.params.items():
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)
            
        for i in range(maxepoch):
            batch_mask = np.random.permutation(np.arange(len(x_train)))
    
            for j in range(len(x_train) // batchsize):
                x_batch = self.x_train[batch_mask[batchsize*j : batchsize*(j+1)]]
                t_batch = self.t_train[batch_mask[batchsize*j : batchsize*(j+1)]]
                grads = self.model.gradient(x_batch, t_batch, self.model.params)
                crt_lr = lr * np.sqrt(1.0 - self.decreace2**(i*batchsize + j+1)) / (1.0 - self.decreace1**(i*batchsize + j+1))
                
                for key in grads.keys():
                    self.m[key] += (1-self.decreace1) * (grads[key]-self.m[key])
                    self.v[key] += (1-self.decreace2) * (grads[key]**2-self.v[key])
                    self.model.params[key] -= crt_lr * self.m[key] / (np.sqrt(self.v[key]) +1e-7)
                
            if (i+1) % check == 0:
                tmp = self.model.accuracy(x_test,t_test)
                print("epoch:",str(i)," | ",str(tmp))
