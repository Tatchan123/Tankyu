import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
import random
import copy


class Toba:
    def __init__ (self,model,x,tobaoption):
        self.model = model
        self.x = x
        self.tobaoption = tobaoption
        
        self.params = copy.deepcopy(model.params)
        self.compare_nodes =eval(tobaoption["rmw_type"])
        self.rmw_layer = tobaoption["rmw_layer"]
        
    def rmw(self):
        for cnt, idx in enumerate(self.rmw_layer):
            x = self.half_predict(idx)
            eps, de = self.tobaoption["epsilon"][cnt], self.tobaoption["delete_n"][cnt]
            rmlist, complist, scalar, bias = self.compare_nodes(x, idx, self.tobaoption, self.params, eps, de)
            self.apply(idx,rmlist, complist, scalar, bias)
            print(idx, "delete", len(rmlist), "nodes")
        return self.params
            
    def half_predict(self, stop_layer):

        batch_x = self.model.predict(self.x,None,False,stop_layer)
        if stop_layer[0] == "C":
            conv_index = stop_layer[-1]
            layer = self.model.layers[stop_layer]
            pad = layer.P
            B,C,Ih,Iw = batch_x.shape
            F = self.params["F"+str(conv_index)]
            M,C,Fh,Fw = F.shape
            Oh = Ih + 2*pad -Fh + 1
            Ow = Iw + 2*pad -Fw + 1
            x = np.pad(batch_x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
            col = layer.im2col(x,B,C,Fh,Fw,Oh,Ow)
            col = col.reshape(C,Fh*Fw,B*Oh*Ow)
            F = np.transpose(F,(1,0,2,3)).reshape(C,M,-1)
            out = []
            for channel in range(C):
                y = np.dot(F[channel],col[channel]).reshape(-1)
                out.append(y)
        else :
            idx = stop_layer[-1]
            out = []
            for i in batch_x:            
                y = (i.reshape(-1,1))*self.params["W"+str(idx)]
                out.append(y)
            out=np.asarray(out)
            out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)
            
        return out
    
    def apply(self, layer, rmlist, complist, scalar, bias):
        if layer[0] == "C":
            conv_index = int(layer[-1])
            self.params["F"+str(conv_index)] = self.params["F"+str(conv_index)] * (scalar.reshape(-1,1,1,1))
            self.params["Cb"+str(conv_index)] = self.params["Cb"+str(conv_index)] + bias
        
            self.params["F"+str(conv_index-1)] = np.delete(self.params["F"+str(conv_index-1)],rmlist,axis=0)
            self.params["Cb"+str(conv_index-1)] = np.delete(self.params["Cb"+str(conv_index-1)],rmlist)
            self.params["F"+str(conv_index)] = np.delete(self.params["F"+str(conv_index)],rmlist,axis=1)
        else:
            idx = int(layer[-1])
            self.params["W"+str(idx)] = self.params["W"+str(idx)] * (scalar.reshape(-1,1))
            self.params["b"+str(idx)] = self.params["b"+str(idx)] + bias
            
            self.params["W"+str(idx-1)] = np.delete(self.params["W"+str(idx-1)],rmlist,axis=1)
            self.params["b"+str(idx-1)] = np.delete(self.params["b"+str(idx-1)],rmlist)
            self.params["W"+str(idx)] = np.delete(self.params["W"+str(idx)],rmlist,axis=0)


def corrcoef(out, layer, tobaoption, params, epsilon, delete_n):
    corlist, rmlist, complist, alist, blist = [], [], [], [], []

    for i in range(len(out) - 1):
        for j in range(i + 1, len(out)):
            i_val, j_val = out[i], out[j]
            sxy = np.mean(i_val * j_val) - np.mean(i_val) * np.mean(j_val)
            vari = np.var(i_val)
            varj = np.var(j_val)
            cor = sxy / (np.sqrt(vari * varj) + 1e-8)  
            a = sxy / (vari + 1e-8)
            b = np.mean(j_val) - a * np.mean(i_val)

            corlist.append(abs(cor))
            rmlist.append(i)
            complist.append(j)
            alist.append(a)
            blist.append(b)

    sorted_data = sorted(zip(corlist, rmlist, complist, alist, blist), reverse=True)
    corlist, rmlist, complist, alist, blist = zip(*sorted_data)

    rmlist_s, complist_s, alist_s, blist_s = [], [], [], []
    for cnt in range(len(rmlist)):
        if len(rmlist_s) >= delete_n:
            break
        if (rmlist[cnt] not in rmlist_s and 
            rmlist[cnt] not in complist_s and 
            corlist[cnt] > epsilon):
            rmlist_s.append(rmlist[cnt])
            complist_s.append(complist[cnt])
            alist_s.append(alist[cnt])
            blist_s.append(blist[cnt])

    alist_s, blist_s = np.array(alist_s), np.array(blist_s)
    pidx = "F" + layer[-1] if layer[0] == "C" else "W" + layer[-1]
    scalar = np.ones(len(params[pidx]))
    scalar[complist_s] += alist_s
    bias = np.sum(blist_s)

    return rmlist_s, complist_s, scalar, bias
