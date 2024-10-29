import jikken.gpu as gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
from collections import OrderedDict
import random

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



    def rmw(self,x,epsilon,complement,rmw_layer):
        params = self.params
        batch_x = self.layers["toba"].forward(x,params)

        for idx in range(1,max(rmw_layer)+1):
            if idx not in rmw_layer:
                batch_x = self.layers["Affine"+str(idx)].forward(batch_x,params)
                batch_x = self.layers["Activation"+str(idx)].forward(batch_x,params)
                continue
            
            rmlist = []
            complist = []
            difflist = []
            out = []
            for i in batch_x:
                y = (i.reshape(-1,1))*params["W"+str(idx)]
                out.append(y)
            out=np.asarray(out)
            out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #y_batch_x
                
            for i in range(0,len(out)-1):
                if i in complist:
                    continue
                for j in range(i+1,len(out)):
                    diff = out[i] - out[j]
                    disp = np.average(diff**2) - np.average(diff)**2
                    if disp <= epsilon[idx-1]:
                        rmlist.append(i)
                        complist.append(j)
                        difflist.append(np.average(diff))
                        break
            if complement:
                difflist=np.asarray(difflist)
                scalar = np.array([1]*len(params["W"+str(idx)]))
                for n in range(len(rmlist)):
                    scalar[int(complist[n])] += scalar[int(rmlist[n])]
                params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
                params["b"+str(idx)] += np.ones(len(params["b"+str(idx)])) * np.sum(difflist)
                
            if idx == 1:
                params["init_remove"].append(rmlist)
                params["W1"] = np.delete(params["W1"],rmlist,axis=0)
            else:
                params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist,axis=1)
                params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist)
                params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist,axis=0)
                
            print("hidden_layer"+str(idx),": delete",str(len(rmlist))+"nodes")
            if idx == max(rmw_layer) : break
            batch_x = np.delete(batch_x,rmlist,axis=1)
            batch_x = self.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = self.layers["Activation"+str(idx)].forward(batch_x,params) 
                          
        return params

    def random_rmw(self,x,epsilon,complement,rmw_layer,delete_n):
        params = self.params
        for idx in rmw_layer:
            lst = random.sample(list(range(len(params["W"+str(idx)]))), delete_n[idx-1])
            if idx == 1:
                params["W1"] = np.delete(params["W1"],lst,axis=0)
                params["init_remove"].append(lst)
            
            else:    
                params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],lst,axis=1)
                params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],lst)
                params["W"+str(idx)] = np.delete(params["W"+str(idx)],lst,axis=0)
                
        return params


    def count_rmw(self,x,epsilon,complement,rmw_layer,rmw_n):
        params = self.params
        batch_x = self.layers["toba"].forward(x,params)

        for idx in range(1,max(rmw_layer)+1):
            if idx not in rmw_layer:
                batch_x = self.layers["Affine"+str(idx)].forward(batch_x,params)
                batch_x = self.layers["Activation"+str(idx)].forward(batch_x,params)
                continue
            out = []
            for i in batch_x:
                y = (i.reshape(-1,1))*params["W"+str(idx)]
                out.append(y)
            out=np.asarray(out)
            out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #y_batch_x
            
            pair_list = []
            displist = []
            
            rmlist = []
            complist = []
            difflist = []
            
            for i in range(0,len(out)-1):
                for j in range(i+1,len(out)):
                    diff = out[i] - out[j]
                    disp = np.average(diff**2) - np.average(diff)**2
                    pair_list.append([i,j])
                    displist.append(disp)
            ziplist = zip(displist,pair_list)
            zipsort = sorted(ziplist)
            displist,pair_list = zip(*zipsort)

            rmpair = [pair_list[0]]
            cnt = 1
            while len(rmpair)<rmw_n:
                for i in rmpair:
                    if i[1] == pair_list[cnt][0] or i[0] == pair_list[cnt][0]:
                        cnt += 1
                        break
                else:
                    rmpair.append(pair_list[cnt])
                    cnt +=1

            
            rmlist = np.array(rmpair).T[0]
            complist = np.array(rmpair).T[1]
            difflist = out[rmlist] - out[complist]   
            difflist = np.mean(difflist,axis=1)    
            if complement:
                difflist=np.asarray(difflist)
                scalar = np.array([1]*len(params["W"+str(idx)]))
                for n in range(len(rmlist)):
                    scalar[int(complist[n])] += scalar[int(rmlist[n])]
                params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
                diffs = np.sum(difflist)
                params["b"+str(idx)] += np.array([np.sum(difflist)]*len(params["b"+str(idx)]))

            if idx == 1:
                params["init_remove"].append(rmlist)
                params["W1"] = np.delete(params["W1"],rmlist,axis=0)
            else:
                params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist,axis=1)
                params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist)
                params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist,axis=0)
            
            print("hidden_layer"+str(idx),": delete",str(len(rmlist))+"nodes")
            if idx == max(rmw_layer) : break
            batch_x = np.delete(batch_x,rmlist,axis=1)
            batch_x = self.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = self.layers["Activation"+str(idx)].forward(batch_x,params) 
                          
        return params
    
    def auto_epsilon_rmw(self,x,complement,rmw_layer):
        params = self.params
        batch_x = self.layers["toba"].forward(x,params)

        for idx in range(1,max(rmw_layer)+1):
            if idx not in rmw_layer:
                batch_x = self.layers["Affine"+str(idx)].forward(batch_x,params)
                batch_x = self.layers["Activation"+str(idx)].forward(batch_x,params)
                continue
            
            rmlist = []
            complist = []
            difflist = []
            out = []
            for i in batch_x:
                y = (i.reshape(-1,1))*params["W"+str(idx)]
                out.append(y)
            out=np.asarray(out)
            #shape : batch, in_size, out_size
            try:
                epsilon = (len(params["W"+str(idx+1)]) / 2.0) * 1e-3
                #Wの分散の逆数×定数(定数は勘のマジックナンバーなので注意)
            except:
                actual_out = np.sum(out,axis=1).reshape(-1) #batch*out_size
                Sout = np.average(actual_out ** 2) - np.average(actual_out)**2
                epsilon = 1e-3 / Sout
                #次の層がsoftmaxのやつ用　出力の分散の逆数を使う
                
            out = (np.transpose(out,(1,0,2))).reshape(len(out[0]),-1)  #in_size,batch,out_size
    
            for i in range(0,len(out)-1):
                if i in complist:
                    continue
                for j in range(i+1,len(out)):
                    diff = out[i] - out[j]
                    disp = np.average(diff**2) - np.average(diff)**2
                    if disp <= epsilon:
                        rmlist.append(i)
                        complist.append(j)
                        difflist.append(np.average(diff))
                        break
            if complement:
                difflist=np.asarray(difflist)
                scalar = np.array([1]*len(params["W"+str(idx)]))
                for n in range(len(rmlist)):
                    scalar[int(complist[n])] += scalar[int(rmlist[n])]
                params["W"+str(idx)] = params["W"+str(idx)] * (scalar.reshape(-1,1))
                params["b"+str(idx)] += np.array([np.sum(difflist)]*len(params["b"+str(idx)]))
                
            if idx == 1:
                params["init_remove"].append(rmlist)
                params["W1"] = np.delete(params["W1"],rmlist,axis=0)
            else:
                params["W"+str(idx-1)] = np.delete(params["W"+str(idx-1)],rmlist,axis=1)
                params["b"+str(idx-1)] = np.delete(params["b"+str(idx-1)],rmlist)
                params["W"+str(idx)] = np.delete(params["W"+str(idx)],rmlist,axis=0)
                
            print("hidden_layer"+str(idx),": delete",str(len(rmlist))+"nodes")
            if idx == max(rmw_layer) : break
            batch_x = np.delete(batch_x,rmlist,axis=1)
            batch_x = self.layers["Affine"+str(idx)].forward(batch_x,params)
            batch_x = self.layers["Activation"+str(idx)].forward(batch_x,params) 
                          
        return params
        






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