

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


class Relu: #2
    pass


class Affine: #3
    pass


class Softmax_Loss: #4
    pass