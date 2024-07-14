

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
class Softmax_Loss
    損失関数とソフトマックスの順伝播逆伝播()-()
class ImportMnist
    データセット「Mnist」の読み込み (別ファイルでやればよかったな)←別ファイルにしたぞ
実行系
具体的方法は後で考えようかなー
"""


import numpy as np
from collections import OrderedDict
import sys, os
from load import load_mnist
from weight import *
from optimize import* # type: ignore

(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
train_size = x_train[0]



class Trainer: #7
    def __init__(self,layer,weightinit,data_n,max_epoch,optimizer):
        self.model = Network(input_size=784, output_size=10, layer_size=layer, weight_init=weightinit)
        self.x_train = x_train[:data_n]
        self.x_test = x_test[:data_n]
        self.max_epoch = max_epoch
        self.optimizer = optimizer
        
    def fit(self):
        for i in range(self.max_epoch):
            pass                   #14日ここまで バッチの作成をoptimizerに含めるかをめっちゃ迷った 必要になってからでもいいかも（多分すぐ必要になる）
        



class Network:
    """
    バッチファイル受け取り
    predict(順伝播)
    loss(損失関数)
    gradient(逆伝播)
    
    実行順・・・gradient(loss(predict))a

    accuracyは別枠ってかんじでsry    
    """
    def __init__ (self, input_size, output_size, layer_size, weight_init, activation="relu"):
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.layer_n = len(self.layer_size)
        self.params = {}
        
        #重み初期化
        wi = weight_init() #クラスのアドレス持って来るつもり（呼び出し元からそのまま持ってくる）
        self.params = wi.weight_initialization(self.input_size, self.layer_size, self.output_size) #一応クラス引き継げそうな感じ 呼び出し確認よろ
        
        #レイヤ初期化
        self.activation = activation
        self.layers = OrderedDict()
        for idx in range(1, self.layer_n+1):
            self.layers["Affine"+str(idx)] = Affine(self.params["W"+str(idx)],  self.params["b" + str(idx)])
            self.layers["Activation"+str(idx)] = self.activation()
        
        idx = self.layer_n + 1        #最終層は上の層と同じくaffine,biasは持つが、reluではなく祖父とマックスなので別で
        self.layers["Affine"+str(idx)] = Affine(self.params["W"+str(idx)],  self.params["b" + str(idx)])
        self.last_layer = Softmax_Loss()
    
    def gradient(self,x,t):
        """
        勾配の算出 呼び出し後loss→predict
        x:入力データ t:正解ラベル
        """
        self.predict(x,t) #損失関数自体はいらないので返り血はうけとらない  各層通過時にレイヤのインスタンスにアクティベーションと重みが保存されるのでそれでok
        
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        #順伝播
        for layers in layers():
            dout = layers.backward(dout)
        #逆伝播
        grads = {}
        for idx in range(1, self.layer_size+2):
            grads["w"+str(idx)] = self.layers["Affine"+str(idx)].dW
            grads["b"+str(idx)] = self.layers["Affine"+str(idx)].db
        
        return grads
        
    def predict(self,y,t):
        for layer in self.layers.values():
            x = layer.forward(x)
        y = self.last_layer.forward(y,t) #softmaxレイヤーのインスタンスを作りたかったためだけに追加 accuracyで使うことも考えxを返り値に
            
        return(x)
    
    def accuracy(self,x,t):
        """
        正確性を求める  多分使わん    作りたかっただけ
        """
        y = self.predict(x)
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
    
    def forward(self,x):
        self.mask = (x <= 0)   #ゼロ以下かどうか
        out = x.copy()
        out[self.mask] = 0     #Trueの項目を0にしているらしい
        return out
    
    def bacward(self,dout):
        dout[self.mask] = 0    #負なら入力値0だから微分値もゼロ、正だと入力＝出力  だと思う・・・
        dx = dout
        return dx



class Affine: #3
    def __init__ (self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self,x):
       self.x = x
       out = np.dot(x,self.W) + self.b
       return out
    
    def backward(self,dout):
        self.dW = np.dot(self.x.T , dout)
        self.db = np.sum(dout , axis=0)
        dx = np.dot(dout,self.W.T)
        return dx


"""
以下大事故
"""
class Softmax_Loss: #4
    def __init__ (self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self, x ,t):
        self.t = t
        if self.y.ndim == 1:
            y = self.y.reshape(1,y.size)
            t = self.t.reshape(1,t.size)
        t = t.argmax(axis=1)
        batch_size = y.shape[0]
        self.loss =  -np.sum(np.log(y[np.arange(self.batch_size),t]+1e-7)) / self.batch_size
        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return batch_size
        
        
        
    # def entropy_error(self,y,t):           交差エントロピー 多分上のミスってるから一応 残しておく
    #     if y.ndim == 1:
    #         t = t.reshape(1,t.size)
    #         y = y.reshape(1,y.size)　　　　13日ここまで~
            
    #     if t.size == y.size:
    #         t = t.argmax(axis=1)
            
    #     self.batch_size = y.shape[0]
    #     return -np.sum(np.log(y[np.arange(self.batch_size),t]+1e-7)) / self.batch_size
            
        