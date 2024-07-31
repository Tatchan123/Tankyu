import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
    
class He:
    def __init__(self):
        self.params = {}
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]
        
        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx-1])
            self.params["W"+str(idx)] = scale * np.random.randn(all_layer[idx-1],all_layer[idx])
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
        
        return self.params
    
class S_random:
    def __init__(self):
        self.params = {}
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]
        
        for idx in range(1,len(all_layer)):
            scale = np.sqrt(9.0/all_layer[idx-1])
            self.params["W"+str(idx)] = scale * np.random.rand(all_layer[idx-1],all_layer[idx])
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])

        
        return self.params


class RC:
    def __init__(self):
        self.params = {}
        self.copies = [10,10,10]      #len(copy)が何個の値をコピーするか、配列の値が各値をそれぞれコピーする回数([2,3,4]なら3つランダムな値を選んで、それぞれ2,3,4回ランダムな場所にコピーする)
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]

        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx])
            self.params["W"+str(idx)] = scale * np.random.randn(all_layer[idx-1],all_layer[idx])
            for jdx in range(1,len(self.copies)):
                line = np.random.randint(1,all_layer[idx-1]) #コピー元の
                row = np.random.randint(1,all_layer[idx])    #アドレス指定
                address = np.random.randint(1,(all_layer[idx-1] * all_layer[idx]),(1,self.copies[jdx]),self.copies[jdx])   #コピー先のアドレス配列生成、調べたら1次元配列の説明しか乗ってなかったので
                np.put(self.params["W"+str(idx)],address,self.params["W"+str(idx)][line][row])

            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
        print(self.params)
        return self.params
    

class Uni_line:
    def __init__(self):
        self.params = {}
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] +  layer + [out]

        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx-1])
            index = scale * np.random.randn(all_layer[idx])
            self.params["W"+str(idx)] = np.tile(index,(all_layer[idx-1],1))
            self.params["b"+str(idx)] = np.zeros(all_layer[idx]) 
        
        return self.params
    
class Uni_row:
    def __init__(self):
        self.params = {}
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]

        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx-1])
            index = scale * np.random.randn(all_layer[idx-1])
            pre_w = np.tile(index,(all_layer[idx],1))
            self.params["W"+str(idx)] = pre_w.T
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
    
        return self.params