import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
import random
import copy
import re

class Toba:
    def __init__(self,model,x,t):
        self.model = model
        self.x = x
        self.t = t
        
    def random_toba(self, rmw_layer, delete_n):
        self.params = copy.deepcopy(self.model.params)
        self.rmw_layer = rmw_layer
        rmlist = {}
        scalar = {}
        bias = {}
        for layer in rmw_layer:
            idx  = int(layer[-1])
            lst = np.random.choice(np.arange(len(self.params["W"+str(idx)])), delete_n[idx-1],replace=False)
            rmlist[layer] = lst
            scalar[layer] = np.ones(len(self.params["W"+str(idx)]))
            bias[layer] = 0
        self.apply(rmlist,scalar,bias)
        return self.params
    
    def coco_sort(self,rmw_layer):
        self.params = copy.deepcopy(self.model.params)
        self.rmw_layer = rmw_layer
        self.corlist, self.rmlist, self.complist, self.alist, self.blist = {},{},{},{},{}
        for layer in self.rmw_layer:
            x = self.half_predict(layer)
            self.corlist[layer], self.rmlist[layer], self.complist[layer], self.alist[layer], self.blist[layer] = self.coco(x)
            print("  ",layer,"done")

    def coco_pick(self,delete_n,epsilon):
        all_rmlist  , all_scalar, all_bias = {},{},{}
        for layer in self.rmw_layer:
            rmlist_s, complist_s, alist_s, blist_s = [], [], [], []
            for cnt in range(len(self.rmlist[layer])):
                if len(rmlist_s) >= delete_n[int(layer[-1])-1]:
                    break
                if self.rmlist[layer][cnt] not in rmlist_s and self.rmlist[layer][cnt] not in complist_s :
                    rmlist_s.append(self.rmlist[layer][cnt])
                    complist_s.append(self.complist[layer][cnt])
                    alist_s.append(self.alist[layer][cnt])
                    blist_s.append(self.blist[layer][cnt])
            alist_s, blist_s = np.array(alist_s), np.array(blist_s)
            scalar = np.ones(len(self.params["W"+(layer[-1])]))
            scalar[complist_s] += alist_s
            bias = np.sum(blist_s)
            
            all_rmlist[layer] = rmlist_s
            all_scalar[layer] = scalar
            all_bias[layer] = bias
        self.apply(all_rmlist,all_scalar,all_bias)
        return self.params


    def apply(self, all_rmlist, all_scalar, all_bias):
        for layer in self.rmw_layer:
            idx = int(layer[-1])
            rmlist, scalar, bias = all_rmlist[layer], all_scalar[layer], all_bias[layer]
            
            if idx == 1:
                self.params["W"+str(idx)] = self.params["W"+str(idx)] * (scalar.reshape(-1,1))
                self.params["b"+str(idx)] = self.params["b"+str(idx)] + bias

                self.model.layers["Toba"].init_remove.append(rmlist)
                self.params["W"+str(idx)] = np.delete(self.params["W"+str(idx)],rmlist,axis=0)
            else:
                self.params["W"+str(idx)] = self.params["W"+str(idx)] * (scalar.reshape(-1,1))
                self.params["b"+str(idx)] = self.params["b"+str(idx)] + bias
                
                self.params["W"+str(idx-1)] = np.delete(self.params["W"+str(idx-1)],rmlist,axis=1)
                self.params["b"+str(idx-1)] = np.delete(self.params["b"+str(idx-1)],rmlist)
                self.params["W"+str(idx)] = np.delete(self.params["W"+str(idx)],rmlist,axis=0)
            
            
    def half_predict(self, stop_layer):
        batch_x = self.model.predict(self.x,self.t,False,stop_layer)
        idx = stop_layer[-1]
        out = []
        for i in batch_x:            
            y = (i.reshape(-1,1))*self.params["W"+str(idx)]
            out.append(y)
        out=np.asarray(out)
        out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)
        return out
    
    def coco(self, out):
        corlist, rmlist, complist, alist, blist = [], [], [], [], []
        for i in range(len(out) - 1):
            print("\r","sorting",str(i)+("/")+str(len(out)-2),end="")
            for j in range(i + 1, len(out)):
                i_val, j_val = out[i], out[j]
                sxy = np.mean(i_val * j_val) - np.mean(i_val) * np.mean(j_val)
                vari = np.var(i_val)
                varj = np.var(j_val)
                cor = sxy / (np.sqrt(vari * varj) + 1e-8)
                a = sxy / (varj + 1e-8)
                b = np.mean(i_val) - a * np.mean(j_val)
    
                corlist.append(abs(cor))
                rmlist.append(i)
                complist.append(j)
                alist.append(a)
                blist.append(b)
        sorted_data = sorted(zip(corlist, rmlist, complist, alist, blist), reverse=True)
        corlist, rmlist, complist, alist, blist = zip(*sorted_data)
        return corlist , rmlist , complist , alist , blist
