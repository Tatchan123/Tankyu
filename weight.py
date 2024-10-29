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
            scale = np.sqrt(100/all_layer[idx-1])
            self.params["W"+str(idx)] = scale * np.random.rand(all_layer[idx-1],all_layer[idx])
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
            print(self.params["W"+str(idx)].shape)
        
        return self.params


class RC:
    def __init__(self):
        self.params = {}
        self.copies = [[78400],[10000],[10000],[1000]]
        #全体の長さがレイヤー数、[100,100,100]はランダムに3つ値をコピーしてそれぞれ100か所ずつ置き換える
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]

        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx])
            self.params["W"+str(idx)] = scale * np.random.randn(all_layer[idx-1],all_layer[idx])
            
            for jdx in range(0,len(self.copies[idx-1])):
                line = np.random.randint(1,all_layer[idx-1]) #コピー元のアドレス指定
                row = np.random.randint(1,all_layer[idx]) 
                address = np.random.randint(1,(all_layer[idx-1] * all_layer[idx]),(self.copies[idx-1][jdx])) 
                #コピー先のアドレス配列生成、調べたら2次元配列も1次元配列でコピー先指定してたので1次元にしてる
                np.put(self.params["W"+str(idx)],address,self.params["W"+str(idx)][line][row])

            self.params["b"+str(idx)] = np.zeros(all_layer[idx])

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
    
class Lines:
    """
self.divsはレイヤー毎に、wを何種類の列にするか(Uni_lineはこれがすべて1の場合に相当する)
長さはレイヤー数
割り切れないことを想定してないので入れる数はニューロン数の約数で頼む
    """
    def __init__(self):
        self.params = {}
        self.divs = [4,4,4,4]
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]

        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx-1])
            index = scale * np.random.randn(all_layer[idx])
            arr = np.tile(index,(int(all_layer[idx-1]/self.divs[idx-1]),1))
            self.params["W"+str(idx)] = arr
            for j in range(0,self.divs[idx-1]-1):
                index = scale * np.random.randn(all_layer[idx])
                arr = np.tile(index,(int(all_layer[idx-1]/self.divs[idx-1]),1))
                self.params["W"+str(idx)] = np.append(self.params["W"+str(idx)],arr,axis=0)
                

            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
            print(self.params["W"+str(idx)].shape)
        return self.params


    
class Uni_row:
    """
    Uni_lineと同じ方法で、最初から90度回転した形でWを生成して、最後に戻してる(Linesも同じ)
    """
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
    
class Rows:
    """
    self.divsの設定方法はLinesと同じ
    """

    def __init__(self):
        self.params = {}
        self.divs = [2,2,2,2]
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]

        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx-1])
            index = scale * np.random.randn(all_layer[idx-1])
            arr = np.tile(index,(int(all_layer[idx]/self.divs[idx-1]),1))
            self.params["W"+str(idx)] = arr
            for j in range(0,self.divs[idx-1]-1):
                index = scale * np.random.randn(all_layer[idx-1])
                arr = np.tile(index,(int(all_layer[idx]/self.divs[idx-1]),1))
                self.params["W"+str(idx)] = np.append(self.params["W"+str(idx)],arr,axis=0)
               
            self.params["W"+str(idx)] = self.params["W"+str(idx)].T
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
            print(self.params["W"+str(idx)].shape)
        return self.params


    
class Ones:    #これはただのゴミ、ちなみに勾配消失してるのでやっぱりゴミ
    def __init__(self):
        self.params = {}
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]

        for idx in range(1,len(all_layer)):
            self.params["W"+str(idx)] = np.ones((all_layer[idx-1],all_layer[idx]))
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
        
        return self.params