import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
import random
import copy

class Network:
    def __init__ (self, input_size, output_size, layer_size, params, toba, activation="relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.layer_n = len(self.layer_size)
        self.params = params
        
        
        #レイヤ初期化
        self.activation = activation
        self.layers = OrderedDict()
        if toba:
            self.layers["toba"] = Toba()
        for idx in range(1, self.layer_n+1):
            self.layers["Affine"+str(idx)] = Affine(idx)
            self.layers["Activation"+str(idx)] = self.activation()
        
        idx = self.layer_n + 1        #最終層は上の層と同じくaffine,biasは持つが、reluではなく祖父とマックスなので別で
        self.layers["Affine"+str(idx)] = Affine(idx)
        self.last_layer = SoftmaxLoss()
    
    
    
    
    def updateparams(self,params):
        self.params = params     
    
    def gradient(self,x,t,params):
        self.params = params
        #順伝播
        self.predict(x,t) #損失関数自体はいらないので返り血はうけとらない  各層通過時にレイヤのインスタンスにアクティベーションと重みが保存されるのでそれでok
        #逆伝播
        self.backward()
            
        grads = {}
        for idx in range(1, self.layer_n+2):
            grads["W"+str(idx)] = self.layers["Affine"+str(idx)].dW
            grads["b"+str(idx)] = self.layers["Affine"+str(idx)].db
        
        return grads
        
    def predict(self,x,t):
        for layer in self.layers.values():
            x = layer.forward(x,self.params)
        y = self.last_layer.forward(x,t) #softmaxレイヤーのインスタンスを作りたかったためだけに追加 accuracyで使うことも考えxを返り値に
            
        return(x)
    
    def backward(self):
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout,self.params)
    
    def accuracy(self,x,t):
        y = self.predict(x,t)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 :  t = np.argmax(t, axis=1)     #大発見 こんな書き方できるのか
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def cal_loss(self,x,t):
        loss = self.predict(x,t)
        return self.last_layer.forward(loss,t)


"""
以下レイヤークラス
"""
class Toba:
    def forward(self,x,params):
        self.x = x
        for i in params["init_remove"]:
            self.x = np.delete(self.x,i,1)
        return self.x
    def backward(self,dout,params):
        return None


        
class Relu:                   
    def __init__ (self):
        self.mask = None
    
    def forward(self,x,params):
        self.mask = (x <= 0)   #ゼロ以下かどうか
        out = x.copy()
        out[self.mask] = 0     #Trueの項目を0にしているらしい
        return out
    
    def backward(self,dout,params):
        dout[self.mask] = 0    #負なら入力値0だから微分値もゼロ、正だと入力＝出力  だと思う・・・
        dx = dout
        return dx



class Affine: #3
    def __init__ (self,idx):
        self.idx = idx
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self,x,params):
        self.x = x
        w = params["W"+str(self.idx)]
        b = params["b"+str(self.idx)]

        out = np.dot(x,w) + b
        return out
    
    def backward(self,dout,params):
        w = params["W"+str(self.idx)]
        b = params["b"+str(self.idx)]
        self.dW = np.dot(self.x.T , dout)
        self.db = np.sum(dout , axis=0)
        dx = np.dot(dout,w.T)
        return dx



class SoftmaxLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmaxの出力
        self.t = None # 教師データ

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = self.cross_entropy_error(self.y, self.t)
        
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
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size