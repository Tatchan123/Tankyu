import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
from data.load import load_mnist
import copy
from trainer import *


class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self,x,params,training=False):
        if training:
            self.mask = np.random.rand(*x.shape) > self.rate
            return x * self.mask
        else:
            return x * (1.0 - self.rate)

    def backward(self, dout, params):
        return dout * self.mask
    
    
    
    
class Relu:                   
    def __init__ (self):
        self.mask = None
    
    def forward(self,x,params,training=False):
        self.mask = (x < 0)
        return np.maximum(0,x)
    
    def backward(self,dout,params):
        dout[self.mask] = 0
        dx = dout
        return dx


class Toba:
    def __init__(self):
        self.init_remove = []

    def forward(self,x,params,training=False):
        if len(self.init_remove) != 0:
            for i in self.init_remove:
                
                x = np.delete(x,i,axis=1)
        return x
    
    def backward(self,dout,params):
        B = dout.shape[0]
        if len(self.init_remove) != 0:
            for i in self.init_remove.reverse():
                zeros = np.zeros((len(i),B,dout.shape[1]))
                dout = np.insert(dout,zeros,i)
        return dout

class Conv2d:
    def __init__(self,idx,pad):
        self.idx = idx
        self.dW = None
        self.db = None
        self.Ishape = None
        self.col = None
        self.P = pad

    def im2col(self,x,B,C,Fh,Fw,Oh,Ow):
        col = np.zeros((B,C,Fh,Fw,Oh,Ow))
        for h in range(Fh):
            for w in range(Fw):
                col[:,:,h,w,:,:] = x[:, :, h:h+Oh, w:w+Ow]
        col = np.transpose(col,(1,2,3,0,4,5))   #(channel,Fh,Fw,batch,Oh,Ow)
        col = np.reshape(col,(C*Fh*Fw,B*Oh*Ow))
        return col
    
    def col2im(self,col,B,C,Fh,Fw,Oh,Ow):
        col = col.reshape(C,Fh,Fw,B,Oh,Ow).transpose(3,0,1,2,4,5)
        im = np.zeros(self.Ishape)
        for h in range(Fh):
            for w in range(Fw):
                im[:,:,h:h+Oh,w:w+Ow] += col[:,:,h,w,:,:]
        return im[:,:, self.P:self.Ishape[2]-self.P,self.P:self.Ishape[3]-self.P]

    def forward(self,x,params,training=False):
        F = params["F"+str(self.idx)].shape
        self.Ishape = x.shape
        B,C,Ih,Iw = x.shape      #batch,channel,imput_height,imput_width
        Fh = F[2]                #filter_height
        Fw = F[3]                #filter_width
        Oh = Ih + 2*self.P -Fh + 1    #output_height
        Ow = Iw + 2*self.P -Fw + 1    #output_width
        M = F[0]                 #filter_n
        
        x = np.pad(x,((0,0), (0,0), (self.P,self.P), (self.P,self.P)),'constant')
        
        self.Ishape = x.shape
        col = self.im2col(x,B,C,Fh,Fw,Oh,Ow)
        # col = np.asarray(col)
        if training:
            self.col = col
        w = params["F"+str(self.idx)].reshape(M,-1)
        b = params["Cb"+str(self.idx)]
        out = np.dot(w,col).reshape(M,B,Oh,Ow)
        out = np.transpose(out,(1,0,2,3)) + b.reshape(1,M,1,1)
        
        return out
    
    def backward(self,dout,params):
        B,M,Oh,Ow = dout.shape
        dout = np.transpose(dout,(1,0,2,3))
        dout = np.reshape(dout,(M,B*Oh*Ow))
        F = params["F"+str(self.idx)].shape
        self.dW = np.dot(dout,self.col.T).reshape(F)
        self.db = np.sum(dout,axis=1)
        dx = np.dot(params["F"+str(self.idx)].reshape(M,-1).T,dout) #shape=(C*Fh*FW,B*Oh*Ow)
        

        C = F[1]
        Fh = F[2]
        Fw = F[3]
        dx = self.col2im(dx,B,C,Fh,Fw,Oh,Ow)

        return dx


class BatchNormalize:
    def __init__(self,idx):
        self.idx = idx
        self.xbn = None #逆伝播に必要
        self.xm = None
        self.var = None
        self.dg = None
        self.db = None
        self.momentum = 0.9

    def forward(self,x,params,training=False):
        if training:
            B,C,h,w = x.shape
            x = np.transpose(x,(0,2,3,1)).reshape(B*h*w,C)
            mu = np.mean(x,axis=0)
            self.xm = x - mu
            self.var = np.var(x,axis=0)
            
            self.xbn = self.xm / np.sqrt(self.var + 1e-7)
            gamma = params["gamma"+str(self.idx)]
            beta = params["beta"+str(self.idx)]
            out = gamma * self.xbn + beta
            out = np.transpose(out.reshape(B,h,w,C),(0,3,1,2))
            
            if np.all(params["move_m"+str(self.idx)]==0):
                params["move_m"+str(self.idx)] = mu
                params["move_v"+str(self.idx)] = self.var
            else:
                params["move_m"+str(self.idx)] = self.momentum * params["move_m"+str(self.idx)] + (1-self.momentum)*mu
                params["move_v"+str(self.idx)] = self.momentum * params["move_v"+str(self.idx)] + (1-self.momentum)*self.var

        else: 
            B,C,h,w = x.shape
            x = np.transpose(x,(0,2,3,1)).reshape(B*h*w,C)
            xbn = (x - params["move_m"+str(self.idx)]) / np.sqrt(params["move_v"+str(self.idx)] + 1e-7)
            gamma = params["gamma"+str(self.idx)]
            beta = params["beta"+str(self.idx)]
            out = gamma * xbn + beta
            out = np.transpose(out.reshape(B,h,w,C),(0,3,1,2))

        return out
    
    def backward(self,dout,params):
        B,C,h,w = dout.shape
        dout = np.transpose(dout,(0,2,3,1)).reshape(B*h*w,C)
        self.db = np.sum(dout, axis=0)
        self.dg = np.sum(dout*self.xbn,axis=0)
        gamma = params["gamma"+str(self.idx)]
        dxbn = dout * gamma

        dvar = np.sum(dxbn * self.xm * -0.5 * ((self.var + 1e-7)**-3/2) , axis=0)
        dxm = dxbn / np.sqrt(self.var + 1e-7) + 2 * self.xm * dvar / (B*h*w)

        dx = dxm - np.mean(dxm,axis=0)
        dx = np.transpose(dx.reshape(B,h,w,C),(0,3,1,2))
        return dx




class Maxpool:
    def __init__(self,idx,hw):
        self.idx = idx
        self.max_index = None
        self.Ishape = None
        self.vpadding = 0
        self.hpadding = 0
        self.Fh = hw
        self.Fw = hw
    def im2col(self,x,B,C,Fh,Fw,Oh,Ow):
        
        col = np.zeros((B,C,Fh,Fw,Oh,Ow))  #(batch,channel,Fh,Fw,Oh,Ow)
        for h in range(Fh):
            for w in range(Fw):
                col[:,:,h,w,:,:] = x[:, :, h:h+2*Oh:2, w:w+2*Ow:2]
        col = np.transpose(col,(0,4,5,1,2,3))   #(batch,Oh,Ow,channel,Fh,Fw)
        col = np.reshape(col,(B*Oh*Ow*C,Fh*Fw))

        return col
    
    def col2im(self,col,B,C,Oh,Ow,Fh,Fw):
        col = col.reshape(B,Oh,Ow,C,Fh,Fw).transpose(0,3,4,5,1,2)  #batch,channel,Fh,Fw,Oh,Ow
        im = np.zeros(self.Ishape)
        for h in range(Fh):
            for w in range(Fw):
                im[:, :, h:h+2*Oh:2, w:w+2*Ow:2] += col[:,:,h,w,:,:]
        im = im[:, :, self.vpadding:, self.hpadding:]
        return im

    def forward(self,x,params,training=False):
        
        B,C,Ih,Iw = x.shape
        
        self.vpadding = (Ih - self.Fh) % 2
        
        self.hpadding = (Iw - self.Fw) % 2
            
        Oh = (Ih + self.vpadding - self.Fh)//2 + 1
        Ow = (Iw + self.hpadding - self.Fw)//2 + 1
        x = np.pad(x,((0,0),(0,0),(self.vpadding,0),(self.hpadding,0)))
        self.Ishape = x.shape
        col = self.im2col(x,B,C,self.Fh,self.Fw,Oh,Ow)
        out = np.max(col,axis=1).reshape(B,Oh,Ow,C).transpose(0,3,1,2)
        if training:
            self.max_index = np.argmax(col,axis=1)
        
        return out
    
    def backward(self,dout,params):
        B,C,Oh,Ow = dout.shape    #Oh,OWは順伝播における出力
        dout = np.transpose(dout,(0,2,3,1)).reshape(-1)
                
        dcol = np.zeros((len(dout),self.Fh*self.Fw))  #B*Oh*Ow*C , Fh*Fw

        # for row,line in enumerate(self.max_index):
        #     dcol[row][line] = dout[row]
        # お前の200万ループをずっと見ていたぞ 本当によく頑張ったな
        dcol[np.arange(B*C*Oh*Ow),self.max_index] = dout
        
        im  = self.col2im(dcol,B,C,Oh,Ow,self.Fh,self.Fw)
        return im
    
class Flatten():
    def __init__(self):
        self.dshape = None
    def forward(self,x,params,training=False):
        B = x.shape[0]
        self.dshape = x.shape
        out = x.reshape(B,-1)
        return out
    def backward(self,dout,params):
        dx = dout.reshape(self.dshape)
        return dx

class Affine: #3
    def __init__ (self,idx):
        self.idx = idx
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self,x,params,training=False):
        if training:
            self.x = x
        w = params["W"+str(self.idx)]
        b = params["b"+str(self.idx)]
        out = np.dot(x,w) + b
        return out
    
    def backward(self,dout,params):
        w = params["W"+str(self.idx)]
        self.dW = np.dot(self.x.T , dout)
        self.db = np.sum(dout , axis=0)
        dx = np.dot(dout,w.T)
        return dx



class SoftmaxLoss:
    def __init__(self,regularize):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ
        self.regularize = regularize
        

    def forward(self, x, t, params):
        self.t = t
        self.y = self.softmax(x)
        if self.regularize is None:
            self.loss = self.cross_entropy_error(self.y, self.t)

        elif self.regularize[0] == "l1": #L1正則化
            regular_term = 0
            for key,prm in params.items():
                if "move" in key:

                    continue
                else:
                    regular_term = np.sum(np.abs(prm))
            self.loss = self.cross_entropy_error(self.y, self.t) + self.regularize[1] * regular_term

        elif self.regularize[0] == "l2": #L2正則化
            regular_term = 0
            for key,prm in params.items():
                if "move" in key:
                    continue
                else:
                    regular_term += np.sum(prm ** 2)
            self.loss = self.cross_entropy_error(self.y, self.t) + self.regularize[1] * regular_term
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
        
    def softmax(self,x):
        x = x - np.max(x, axis=-1, keepdims=True)   # オーバーフロー対策
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
         
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
             
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-15)) / batch_size