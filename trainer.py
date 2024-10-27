import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from model1 import *
from weight import *
import time


class Trainer:
    def __init__(self, step, layer, weightinit, optimizer, data, batch_size, lr, check, decreace1=None, decreace2=None, epsilon=None, complement=None, rmw_layer=None, delete_n=None):
        
        self.step = step
        self.layer = layer
        self.x_train = data["x_train"]
        self.t_train = data["t_train"]
        self.x_test = data["x_test"]
        self.t_test = data["t_test"]
        self.data_n = len(self.x_train)
        self.batch_size = batch_size
        self.lr = lr
        self.check = check
        self.decreace1 = decreace1
        self.decreace2 =decreace2
        self.epsilon = epsilon
        self.complemetnt = complement
        self.rmw_layer = rmw_layer
        self.delete_n = delete_n
        if optimizer == "sgd": self.opt=self.sgd
        if optimizer == "adam": self.opt=self.adam
        
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
                self.params = self.opt(step)
            else:
                if step == "rmw":
                    self.params = self.rmw()
                if step == "random_rmw":
                    self.params = self.random_rmw()
                    
        t1 = time.time()    
        self.model.updateparams(self.params)
        acc = self.model.accuracy(self.x_test,self.t_test)
        t2 = time.time()
        elapsed_time = t2-t1
        return [float(acc),elapsed_time*1000]


    def sgd(self,maxepoch):
        params = self.params
        cnt = 0
        for i in range(maxepoch):
            batch_mask = np.random.choice(self.data_n, self.batch_size,self.params)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            
            grads = self.model.gradient(x_batch, t_batch, params)
            for key in grads.keys():
                params[key] -= self.lr*grads[key]
            
            if cnt == self.check:
                cnt = 0
                tmp = self.model.accuracy(self.x_test,self.t_test)
                print("epoch:",str(i)," | ",str(tmp))
            cnt += 1
        return params
    
    
    def adam(self,maxepoch):
        params = self.params
        cnt = 0
        self.m,self.v = {},{}
        for k,v in params.items():
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)
            
        for i in range(maxepoch):
            batch_mask = np.random.choice(self.data_n,self.batch_size,params)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]
            grads = self.model.gradient(x_batch, t_batch, params)
            crt_lr = self.lr * np.sqrt(1.0 - self.decreace2**(cnt+1)) / (1.0 - self.decreace1**(cnt+1))
            
            for key in grads.keys():
                self.m[key] += (1-self.decreace1) * (grads[key]-self.m[key])
                self.v[key] += (1-self.decreace2) * (grads[key]**2-self.v[key])
                params[key] -= crt_lr * self.m[key] / (np.sqrt(self.v[key]) +1e-7)
                
            if cnt == self.check:
                cnt = 0
                tmp = self.model.accuracy(self.x_test,self.t_test)
                print("epoch:",str(i)," | ",str(tmp))
            cnt += 1        
        return params
    
    
    def rmw(self):
        params = self.params
        print("start rmw ===========================================")
        self.model.updateparams(params)
        print("accuracy before rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        
        params = self.model.rmw(self.x_train,self.epsilon,self.complemetnt,self.rmw_layer)
        
        self.model.updateparams(params)
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.layer)+2)):
            tmp = np.append(tmp,params["b"+str(i)].shape)
        print("Composition of Network :",tmp)
        print("accuracy after rmw :",str(self.model.accuracy(self.x_test,self.t_test)))
        print("finish rmw ==========================================")
        return params
    
    
    def random_rmw(self):
        params = self.params
        print("RANDOM R M W !!!!!!!!!!!!!")
        params = self.model.random_rmw(self.x_train, self.epsilon, self.complemetnt, self.rmw_layer, delete_n=self.delete_n)
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.layer)+2)):
            tmp = np.append(tmp,params["b"+str(i)].shape)
        print("Composition of Network :",tmp)
        return params
    
    def measure(self):
        t1 = time.time()
        self.model.updateparams(self.params)
        acc = self.model.accuracy(self.x_test,self.t_test)
        t2 = time.time()
        elapsed_time = t2-t1
            