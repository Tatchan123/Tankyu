

"""
class Trainer
    バッチ作成、学習の進行 ()-()  optimizerはこちらです(Networkじゃない) ←わかりにくくなったら変える
class Network 
    学習、重み計算 ()-()
class Optimizer          長くなりすぎたらわんちゃん消すかも←消去済み 同ディレクトリoptimizerへ
    最適化 ()-()


class Relu
    relu順伝播、逆伝播 ()-()
class Affine
    全結合層順伝播、逆伝播 ()-()
class SoftmaxLoss
    損失関数とソフトマックスの順伝播逆伝播()-()
class ImportMnist
    データセット「Mnist」の読み込み (別ファイルでやればよかったな)←別ファイルにしたぞ
実行系
具体的方法は後で考えようかなー
"""


import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
from collections import OrderedDict
import copy


class Toba:
    def __init__(self):
        self.init_remove = []
        
        
        
    def rmw(self, x, params, epsilon, complement,rmw_layer):
        """
        idx:trainer側でレイヤー数この関数を繰り返すので層数も引数にとる
        epsilon:分散がこの値より小さいときニューロンを結合する float?
        complement:Trueならニューロン同士の特徴量の差を補完して削除する側に足す Falseなら何もしない
        """
        self.params = copy.deepcopy(params)
        x = self.forward(x,self.params)
        for idx in range(1,max(rmw_layer)+1):
            if idx not in rmw_layer:
                x = np.dot(x,params["W"+str(idx)]) + params["b"+str(idx)]
                continue
            w = self.params["W"+str(idx)]
            out = []
            for i in x:
                y = (i.reshape(-1,1))*w
                out.append(y)
            out = np.asarray(out)
            out = (np.transpose(out, (1, 0, 2))).reshape(len(out[0]),-1)
            rmlist = []
            completionlist = np.empty((0,2),dtype=int)
            difflist = np.array([])
            
            for i in range(0,len(out)-1):
                for j in range(i+1,len(out)):
                    
                    diff = out[i] - out[j]
                    disp = np.average(diff ** 2) - np.average(diff) ** 2
                    #分散 = 2乗の平均 - 平均の2乗
                    #print(disp)
                    if disp <= epsilon[idx-1]:
                        one_comp = np.array([j,1],dtype=int)
                        rmlist.append(i)
                        #diffはjが依存先になっている数だけ倍にする iが依存先になっているのをみつけて、そいつが依存先になっている数を足す
                        for sublist in completionlist:
                            if i == sublist[0]:
                                one_comp[1] += sublist[1]
                        completionlist = np.vstack((completionlist,one_comp))
                        difflist = np.append(difflist,np.average(diff)*one_comp[1])
        
                        break
            rmlist = np.asarray(rmlist,dtype='int32')
            if complement:
                scalar = np.ones(len(self.params["W"+str(idx)]))
                for i in range(len(rmlist)):
                    scalar[int(completionlist[i][0])] += scalar[int(rmlist[i])]
                print(rmlist)
                print(completionlist)
                self.params["W"+str(idx)] = self.params["W"+str(idx)] * (scalar.reshape(-1,1))
                self.params["b"+str(idx)] += np.sum(difflist)
            
            if idx == 1:
                self.init_remove.append(rmlist)
                self.params["W"+str(idx)] = np.delete(self.params["W"+str(idx)],rmlist,axis=0)
            else:
                self.params["W"+str(idx)] = np.delete(self.params["W"+str(idx)],rmlist,axis=0)
                self.params["W"+str(idx-1)] = np.delete(self.params["W"+str(idx-1)],rmlist,axis=1)
                self.params["b"+str(idx-1)] = np.delete(self.params["b"+str(idx-1)],rmlist,0)
                
            print("hidden_layer"+str(idx),": delete",str(len(rmlist))+"nodes")
            
            x = np.delete(x,rmlist,1)
            x = np.dot(x,self.params["W"+str(idx)]) + self.params["b"+str(idx)]
                
        return self.params
    
    def rmw_random(self,params,deleat_n,rmw_layer):
        new_params = copy.deepcopy(params)

        if 1 in rmw_layer:
            lst = np.random.choice(np.arange(len(params["W1"])), deleat_n[1])
            self.init_remove.append(lst)
            new_params["W1"] = np.delete(new_params["W1"],lst,axis=0)
            #一応入力削るのに対応してるけどそもそもjikken1のモデルにtobaレイヤーないのでどっちにしろacc()でエラー
            
        for i in range(2,max(rmw_layer)+1):
            lst = np.random.choice(np.arange(len(new_params["W"+str(i)])), deleat_n[i])
            new_params["W"+str(i)]  = np.delete(new_params["W"+str(i)],lst,axis=0)
            new_params["W"+str(i-1)] = np.delete(new_params["W"+str(i-1)],lst,axis=1)
            new_params["b"+str(i-1)] = np.delete(new_params["b"+str(i-1)],lst, axis=0)

        
        return new_params

    
    def forward(self,x,params):
        if not self.init_remove==[]:
            for i in self.init_remove:
                x = np.delete(x,i,1)
        return x
            
    def backward(self,dout,params):
        return None


class Network:
    """
    バッチファイル受け取り
    predict(順伝播)
    loss(損失関数)
    gradient(逆伝播)
    
    """
    def __init__ (self, input_size, output_size, layer_size, params, activation="relu", toba=False):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.layer_n = len(self.layer_size)
        self.params = params
        self.rmlist = []
        
        
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
    
    
    
    
    
    
    
    def gradient(self,x,t,params):
        self.params = params
        """
        勾配の算出 呼び出し後loss→predict
        x:入力データ t:正解ラベル
        """
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
        for k,layer in self.layers.items():
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

    def updateparams(self,params):
        self.params = params 










"""
   以下レイヤーのクラス めんどいので直に書いた  
   x入力はバッチ全体の画像データ [[画像2]、[画像4],[画像10],・・・
"""




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

class Identity: #恒等関数(y=x)
    def __init__(self):
        pass
    def forward(self,x,params):
        out = x.copy()
        return out
    
    def backward(self,dout,params):
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


"""
以下大事故
"""
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
        