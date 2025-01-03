


from gpu import *
if Use_Gpu:
    import cupy as np
else:
    import numpy as np
#padとかスライスでcupyがなんか事故るので両方いれる
    
from collections import OrderedDict
import copy
from layer import *


class Convnetwork:

    def __init__ (self, input_size, output_size, dense_layer, weightinit, conv_layer=[],activation=Relu,batchnorm=True, toba=False, drop_rate=[0,0], regularize=None):
        self.input_size = input_size
        self.output_size = output_size
        self.dense_layer = dense_layer
        self.conv_layer = conv_layer
        self.layer_n = len(self.dense_layer)
        self.batchnorm = batchnorm
        self.regularize = regularize
        
        #try:
        wi = weightinit()
        self.params = wi.weight_initialization(inp=input_size,layer=dense_layer,convlayer=conv_layer,out=10,batchnorm=self.batchnorm)
        #except:
        #    self.params = weightinit
        
            
        #レイヤ初期化
        self.activation = activation
        self.layers = OrderedDict()
        cc = 1
        pc = 1
        for idx in range(1, len(self.conv_layer)+1):
            
            if len(self.conv_layer[idx-1]) == 3:
                
                self.layers["Conv2d"+str(cc)] = Conv2d(cc,conv_layer[idx-1][2])
                if self.batchnorm:
                    self.layers["BatchNorm"+str(cc)] = BatchNormalize(cc)
                if drop_rate[0] != 0:
                    self.layers["ConvDrop"+str(cc)] = Dropout(drop_rate[0])
                self.layers["ConvActivation"+str(cc)] = self.activation()
                
                cc += 1
            elif len(self.conv_layer[idx-1]) == 1:    
                self.layers["Maxpool"+str(pc)] = Maxpool(pc,conv_layer[idx-1][0])
                pc += 1


        self.layers["Flatten"] = Flatten()
        if toba:
            self.layers["Toba"] = Toba()


        for idx in range(1, self.layer_n+1):
            self.layers["Affine"+str(idx)] = Affine(idx)
            if drop_rate[1] != 0:
                self.layers["Dropout"+str(idx)] = Dropout(drop_rate[1])
            self.layers["Activation"+str(idx)] = self.activation()
        
        idx = self.layer_n + 1        #最終層は上の層と同じくaffine,biasは持つが、reluではなく祖父とマックスなので別で
        self.layers["Affine"+str(idx)] = Affine(idx)
        self.last_layer = SoftmaxLoss(regularize)
    
    
    def gradient(self,x,t,params):
        self.params = params
        """
        勾配の算出 呼び出し後loss→predict
        x:入力データ t:正解ラベル
        """
        #順伝播
        self.predict(x,t,True) #損失関数自体はいらないので返り血はうけとらない  各層通過時にレイヤのインスタンスにアクティベーションと重みが保存されるのでそれでok
        #逆伝播
        self.backward()
            
        grads = {}
        cc = 1   #番号を畳み込み、プーリング、全結合で別々に振ってるのでfor i すると番号が被る
        ac = 1
        if self.regularize is None:
            for layer_name in self.layers.keys():
                if layer_name == "Conv2d"+str(cc):
                    grads["F"+str(cc)] = self.layers[layer_name].dW
                    grads["Cb"+str(cc)] = self.layers[layer_name].db
                    if self.batchnorm:
                        grads["gamma"+str(cc)] = self.layers["BatchNorm"+str(cc)].dg
                        grads["beta"+str(cc)] = self.layers["BatchNorm"+str(cc)].db
                    cc += 1
                
                elif layer_name == "Affine"+str(ac):
                    grads["W"+str(ac)] = self.layers["Affine"+str(ac)].dW
                    grads["b"+str(ac)] = self.layers["Affine"+str(ac)].db
                    ac += 1

        elif self.regularize[0] == "l1":
            alpha = self.regularize[1]
            for layer_name in self.layers.keys():
                if layer_name == "Conv2d"+str(cc):
                    grads["F"+str(cc)] = self.layers[layer_name].dW + alpha * np.where(params["F"+str(cc)] < 0,-1,np.where(params["F"+str(cc)] > 0,1,0))
                    #パラメータの値が負ならば-1,正ならば1,ゼロならば0を出す
                    grads["Cb"+str(cc)] = self.layers[layer_name].db + alpha * np.where(params["Cb"+str(cc)] < 0,-1,np.where(params["Cb"+str(cc)] > 0,1,0))
                    if self.batchnorm:
                        grads["gamma"+str(cc)] = self.layers["BatchNorm"+str(cc)].dg + alpha * np.where(params["gamma"+str(cc)] < 0,-1,np.where(params["gamma"+str(cc)] > 0,1,0))
                        grads["beta"+str(cc)] = self.layers["BatchNorm"+str(cc)].db + alpha * np.where(params["beta"+str(cc)] < 0,-1,np.where(params["beta"+str(cc)] > 0,1,0))
                    cc += 1
                
                
                elif layer_name == "Affine"+str(ac):
                    grads["W"+str(ac)] = self.layers["Affine"+str(ac)].dW + alpha * np.where(params["W"+str(cc)] < 0,-1,np.where(params["W"+str(ac)] > 0,1,0))
                    grads["b"+str(ac)] = self.layers["Affine"+str(ac)].db + alpha * np.where(params["b"+str(cc)] < 0,-1,np.where(params["b"+str(ac)] > 0,1,0))
                    ac += 1
        
        elif self.regularize[0] == "l2":
            alpha = self.regularize[1]
            for layer_name in self.layers.keys():
                if layer_name == "Conv2d"+str(cc):
                    grads["F"+str(cc)] = self.layers[layer_name].dW + 2 * alpha * params["F"+str(cc)]
                    grads["Cb"+str(cc)] = self.layers[layer_name].db + 2 * alpha * params["Cb"+str(cc)]
                    if self.batchnorm:
                        grads["gamma"+str(cc)] = self.layers["BatchNorm"+str(cc)].dg + 2 * alpha * params["gamma"+str(cc)]
                        grads["beta"+str(cc)] = self.layers["BatchNorm"+str(cc)].db + 2 * alpha * params["beta"+str(cc)]
                    cc += 1
                
                elif layer_name == "Affine"+str(ac):
                    grads["W"+str(ac)] = self.layers["Affine"+str(ac)].dW + 2 * alpha * params["W"+str(ac)]
                    grads["b"+str(ac)] = self.layers["Affine"+str(ac)].db + 2 * alpha * params["b"+str(ac)]
                    ac += 1
        
        return grads
        
    def predict(self,x,t,training):
        for layer in self.layers.values():
            
            x = layer.forward(x,self.params,training)
            # print(layer)
            # print(x.shape)
        y = self.last_layer.forward(x,t,self.params) #softmaxレイヤーのインスタンスを作りたかったためだけに追加 accuracyで使うことも考えxを返り値に
        return(x)
    
    
    def backward(self):
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            #print(layer)
            dout = layer.backward(dout,self.params)
            


    def accuracy(self,x,t):
        y = self.predict(x,t,training=False)

        y = np.argmax(y,axis=1)
        if t.ndim != 1 :  t = np.argmax(t, axis=1)     #大発見 こんな書き方できるのか
        accuracy = np.sum(y==t) / float(x.shape[0])
        return accuracy

    def cal_loss(self,x,t):
        loss = self.predict(x,t,training=False)
        return self.last_layer.forward(loss,t,self.params)


    def updateparams(self,params):
        self.params = params


        
