

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


import numpy as np
from collections import OrderedDict
import sys, os
#from weight import *
# from optimize import* # type: ignore

# (x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)



# class Trainer: #7
#     def __init__(self,layer,weightinit,data_n,max_epoch,optimizer,batch_size):
#         self.train_size = x_train.shape[0]                     #全体の画像の数
#         self.model = Network(input_size=784, output_size=10, layer_size=layer, weight_init=weightinit)
#         self.x_train = x_train[:data_n]
#         self.x_test = x_test[:data_n]
#         self.max_epoch = max_epoch
#         self.optimizer = optimizer
#         self.batch_size = batch_size
        
#     def fit(self):
#         for i in range(self.max_epoch):
#             batch_mask = np.random.choice(self.train_size,self.batch_size)                   #14日ここまで バッチの作成をoptimizerに含めるかをめっちゃ迷った 必要になってからでもいいかも（多分すぐ必要になる）
#             x_batch = x_train[batch_mask]
#             t_batch = t_train[batch_mask]
            
#             grads = self.model.gradient(x_batch,t_batch)
#             self.model.params = self.optimizer.update(self.model.params,grads)



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
    
    def accuracy(self,x,t):
        """
        正確性を求める  多分使わん    作りたかっただけ
        """
        y = self.predict(x,t)
        y = np.argmax(y,axis=1)
        if t.ndim != 1 :  t = np.argmax(t, axis=1)     #大発見 こんな書き方できるのか
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy















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
    # def entropy_error(self,y,t):           交差エントロピー 多分上のミスってるから一応 残しておく
    #     if y.ndim == 1:
    #         t = t.reshape(1,t.size)
    #         y = y.reshape(1,y.size)　　　　13日ここまで~
            
    #     if t.size == y.size:
    #         t = t.argmax(axis=1)
            
    #     self.batch_size = y.shape[0]
    #     return -np.sum(np.log(y[np.arange(self.batch_size),t]+1e-7)) / self.batch_size
            
        