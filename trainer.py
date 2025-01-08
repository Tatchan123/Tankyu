import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
import time
import sys
from Toba_w import *

class Trainer:
    def __init__(self, model, step, optimizer, data, check, tobaoption=None,scheduler=None):
        
        self.step = step
        self.x_train = data["x_train"]
        self.t_train = data["t_train"]
        self.x_test = data["x_test"]
        self.t_test = data["t_test"]
        self.data = data
        self.batch_size = optimizer["batch_size"]
        self.check = check
        self.tobaoption = tobaoption
        
        self.model = model
        self.optimizer = Optimizer(self.model,optimizer, scheduler)
        
    def fit(self):
        print("start")
        result = None
        for step in self.step:
            if type(step) == int:
                self.optimizer.fit(self.data,self.batch_size,step,self.check)
            else:
                tobafunc = eval(step)
                tobaname = step
                params,result = self.toba_w(tobafunc,tobaname)
                
        
        print("finish")
        return result


    def toba_w(self,toba_type,toba_name):
        print("start Toba_W :",toba_name,"------------------------------------------")
        acc1 = self.model.accuracy(self.x_test,self.t_test)
        print("    accuracy before Toba_W :", str(acc1))
        #params = toba_type(self.model,self.x_test,self.tobaoption)
        #self.model.updateparams(params)
        tobaclass = Toba(self.model, self.x_test, self.tobaoption)
        params = tobaclass.rmw()
        self.model.updateparams(params)
        tmp = [params["W1"].shape[0]]
        for i in range(1,int(len(self.model.dense_layer)+2)):
            tmp = np.append(tmp,self.model.params["b"+str(i)].shape)
        print("    Composition of Network :",tmp)
        acc2 = self.model.accuracy(self.x_test,self.t_test)
        print("    accuracy after rmw :",str(acc2))
        print("finish Toba_W ------------------------------------------")
        return self.model.params, {"dacc":acc2-acc1,"acc":acc2}
        
        
        

class Optimizer():
    def __init__(self,model,opt_dict, scheduler=None):
        self.lr = opt_dict["lr"]
        self.model = model
        opt = opt_dict["opt"]
        if opt == "sgd":
            self.fit = self.SGD
        elif opt == "adam":
            self.fit = self.Adam
            self.decreace1 = opt_dict["dec1"]
            self.decreace2 = opt_dict["dec2"]
            
            self.schedule = Scheduler(model,scheduler,self.lr).schedule


    def SGD(self, data,batchsize,maxepoch,check):
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
                    self.model.params[key] -= self.lr*grads[key]
                
            if (i+1) % check == 0:
                acc = self.model.accuracy(x_test,t_test)
                print("epoch:",str(i)," |  accuracy:"+str(acc),"loss:"+str(self.model.cal_loss(x_test,t_test)))
        
    def Adam(self,data,batchsize,max_epoch,check):
        x_train = data["x_train"]
        t_train = data["t_train"]
        x_test = data["x_test"]
        t_test = data["t_test"]
        self.m,self.v = {},{}
        for k,v in self.model.params.items():
            self.m[k] = np.zeros_like(v)
            self.v[k] = np.zeros_like(v)

        for i in range(0,max_epoch):
            lr = self.schedule(i)
            
            batch_mask = np.random.permutation(np.arange(len(x_train)))
            

            for j in range(len(x_train) // batchsize):
                x_batch = x_train[batch_mask[batchsize*j : batchsize*(j+1)]]
                t_batch = t_train[batch_mask[batchsize*j : batchsize*(j+1)]]

                
                grads = self.model.gradient(x_batch, t_batch, self.model.params)

                for key in self.model.params.keys():
                    
                    if "move" in key:
                        continue
                    else:
                        self.m[key] = (1-self.decreace1) * grads[key] + self.decreace1 * self.m[key]
                        self.v[key] = (1-self.decreace2) * grads[key]**2 + self.decreace2 * self.v[key]
                        
                        ctr_lr = lr * np.sqrt(1.0-self.decreace2**(i*batchsize + j+1)) / (1.0-self.decreace1**(i*batchsize + j+1))

                        self.model.params[key] -= ctr_lr * self.m[key] / (np.sqrt(self.v[key]) +1e-7)

                #print(self.model.cal_loss(x_test,t_test))
            if (i+1) % check ==0:
                acc = self.model.accuracy(x_test,t_test)
                loss = self.model.cal_loss(x_test,t_test)
                print("epoch:",str(i+1)," |  accuracy:"+str(acc),"loss:"+str(loss))
                print("traindata  |  accuracy:"+str(self.model.accuracy(x_batch,t_batch)),"loss:"+str(self.model.cal_loss(x_batch,t_batch)))
                print("---------------------------------------------------------------")

        
class Scheduler():
    """
    学習率スケジューラ
    Pytorchのやつをまんま真似してる
    mgn: これにlrをかけたものが最終的な学習率になる

    CosineAnnealing: cos関数で周期的に変化
        min_mgn: mgnの最小値
        T_max: 何エポックで最小値になるか(=周期の半分)
    Exponential: 指数関数
        base: 指数関数の底
    MultiStep: 指定のエポックになったら学習率を減衰
        milestones: 学習率を減衰するタイミングのリスト e.g.[20,40,60,80]
        gamma: 減衰率
    Identity: 等倍、つまりなにもしない scheduler=Noneのときは自動的にこれ
    """
    def __init__(self,model,scheduler,lr): #引数随時追加しなくていいように辞書で渡すようにした
        self.model = model
        self.lr = lr
        if scheduler is None:
            self.schedule = self.Identity
        else:
            method = scheduler["method"]
            if method == "cosine":
                self.schedule = self.CosineAnnealing
                self.min_mgn = scheduler["min_mgn"]
                self.T_max = scheduler["T_max"]
            
            elif method == "exp":
                self.schedule = self.Exponential
                self.base = scheduler["base"]

            elif method == "multistep":
                self.schedule = self.MultiStep
                self.milestones = scheduler["milestones"]
                self.gamma = scheduler["gamma"]
    
    def CosineAnnealing(self,epoch):
        mgn = (1-self.min_mgn)/2 * np.cos(np.pi*epoch/self.T_max) + (1+self.min_mgn)/2
        return self.lr * mgn

    def Exponential(self,epoch):
        mgn = self.base ** epoch
        return self.lr * mgn

    def MultiStep(self,epoch):
        count = 0
        for i in self.milestones:
            if epoch < i:
                break
            count += 1
        mgn = self.gamma **count
        return self.lr * mgn


    def Identity(self,epoch):
        return self.lr
