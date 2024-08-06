

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



class Network:
    """
    バッチファイル受け取り
    predict(順伝播)
    loss(損失関数)
    gradient(逆伝播)
    
    実行順・・・gradient(loss(predict))a

    accuracyは別枠ってかんじでsry    
    """
    def __init__ (self, input_size, output_size, layer_size, params, activation="relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.layer_n = len(self.layer_size)
        self.params = params
        
        #重み初期化
        """
        wi = weight_init() #クラスのアドレス持って来るつもり（呼び出し元からそのまま持ってくる）
        self.params = wi.weight_initialization(self.input_size, self.layer_size, self.output_size) #一応クラス引き継げそうな感じ 呼び出し確認よろ
        """
        
        #レイヤ初期化
        self.activation = activation
        self.layers = OrderedDict()
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


    def rmw(self, idx, x, epsilon, complement):
        """
        idx:trainer側でレイヤー数この関数を繰り返すので層数も引数にとる
        epsilon:分散がこの値より小さいときニューロンを結合する float?
        complement:Trueならニューロン同士の特徴量の差を補完して削除する側に足す Falseなら何もしない
        """
        out = Affine(idx).testward(self,x,self.params)
        rmlist = []
        for i in range(0,len(out[0])):    # 全パターン試すためのfor i,for j
            for j in range(i+1,len(out[0])):
                diff = out[0][i] - out[0][j]
                for k in range(1,len(out)): # バッチ全部の差をとるためのfor k
                    diff = np.append(diff,out[k][i] - out[k][j])
                disp = (np.average((diff ** 2))) - (np.average(diff) ** 2)
                #分散 = 2乗の平均 - 平均の2乗
                if disp <= epsilon:
                    rmlist.append(j)
        
        for i in range(0,len(rmlist)):
            for j in range(i+1,len(rmlist)):
                if rmlist[i] == rmlist[j]:
                    rmlist = np.delete(rmlist,j,axis=0)
        
        if complement:
            pass # 工事中 一旦スルーで
        else:
            self.params["W"+str(idx)] = np.delete(self.params["W"+str(idx)],rmlist,axis=0)
            self.params["W"+str(idx-1)] = np.delete(self.params["W"+str(idx)],rmlist,axis=1)
        
        return self.params["W"+str(idx)], self.params["W"+str(idx-1)]

    def accuracy(self,x,t):
        """
        正確性を求める  多分使わん    作りたかっただけ
        """
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
    
    def testward(self,x,params):
        self.x = x
        w = params["W"+str(self.idx)]
        
        for i in range(0,len(x)):
            row = np.multiply(x[i][0],w[0])
            for j in range(1,len(x[i])):
                row = np.vstack([row,np.multiply(x[i][j],w[j])])
            
            if i == 0:
                out = [row]
            else:
                out = np.append(out,[row],axis=0)
        return out.T


    
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
    # def entropy_error(self,y,t):           交差エントロピー 多分上のミスってるから一応 残しておく
    #     if y.ndim == 1:
    #         t = t.reshape(1,t.size)
    #         y = y.reshape(1,y.size)　　　　13日ここまで~
            
    #     if t.size == y.size:
    #         t = t.argmax(axis=1)
            
    #     self.batch_size = y.shape[0]
    #     return -np.sum(np.log(y[np.arange(self.batch_size),t]+1e-7)) / self.batch_size
            
        