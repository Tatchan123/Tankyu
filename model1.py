

"""
Class Network 
    学習、重み計算 ()-()
class Optimizer
    最適化 ()-()
class Relu
    relu順伝播、逆伝播 ()-()
class Affine
    全結合層順伝播、逆伝播 ()-()
class Softmax_Loss
    損失関数とソフトマックスの順伝播逆伝播()-()
class ImportMnist
    データセット「Mnist」の読み込み (別ファイルでやればよかったな)
実行系
具体的方法は後で考えようかなー
"""


import numpy as np
from collections import OrderedDict
import sys, os
from load import load_mnist


(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True)
train_size = x_train[0]
max_interaction = 2000
print(x_train)
a = "aaaa"


class Fit: #7
    pass



class Network: #6
    pass



class Optimizer: #5
    pass



class Relu:                   #　x入力はバッチ全体の画像データ [[画像2]、[画像4],[画像10],・・・
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
        return dout



class Affine: #3
    def __init__ (self,W,b):
        self.W = W
        self.b = b
        self.x = None
        self.ori_x_shape = None
        self.dW = None
        self.db = None
        
    def forward(self,x):
        self.ori_x_shape = x.shape
        self.x = x.reshape(x.shape[0],-1)   #　xの次元を反転　ん難解********要確認
        
        


class Softmax_Loss: #4
    pass