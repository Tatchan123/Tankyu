import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from model1 import *
from weight import *
import time
import sys
from Toba_w import *

class Trainer:
    def __init__(self, step, layer, weightinit, optimizer, data, batch_size, lr, check, tobaoption):
        
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
        self.tobaoption = tobaoption
        
        try:
            wi = weightinit()
            self.params = wi.weight_initialization(inp=784, layer=layer, out=10)
        except:
            self.params = weightinit
        self.params["init_remove"] = np.array([])
        self.model = Network(input_size=784, output_size=10, layer_size=layer, params=self.params, activation=Relu, toba=True)
        self.optimizer = Optimizer(self.model,optimizer)
        
    def fit(self):
        for step in self.step:
            if type(step) == int:
                self.optimizer.fit(self.params,self.data,self.batch_size,step,self.lr,self.check)
            else:
                tobafunc = eval(step)
                tobaname = step
                self.params,result = self.toba_w(tobafunc,tobaname)
                
        t1 = time.time()
        t2 = time.time()
        elapsed_time = t2-t1
        result["time"] = elapsed_time
        return result


    def toba_w(self,toba_type,toba_name):
        params = self.params
        print("start Toba_W :",toba_name,"------------------------------------------")
        self.model.updateparams(params)
        acc1 = self.model.accuracy(self.x_test,self.t_test)
        print("    accuracy before Toba_W :", str(acc1))
        
        params = toba_type(self.model,self.x_train,self.tobaoption)
        self.model.updateparams(params)
        
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.layer)+2)):
            tmp = np.append(tmp,params["b"+str(i)].shape)
        print("    Composition of Network :",tmp)
        acc2 = self.model.accuracy(self.x_test,self.t_test)
        print("    accuracy after rmw :",str(acc2))
        print("finish Toba_W ------------------------------------------")
        return params, {"dacc":acc2-acc1,"acc":acc2}
        
        
        

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


    def SGD(self,params, data,batchsize,maxepoch,lr,check):
        self.model.updateparams(params)
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
        return self.model.params
    
        
    def Adam(self,params,data,batchsize,maxepoch,lr,check):
        self.model.updateparams(params)
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

            return self.model.params