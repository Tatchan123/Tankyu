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
    
class S_random:
    def __init__(self):
        self.params = {}
    def weight_initialization(self, inp, layer, out):
        all_layer = [inp] + layer + [out]
        
        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx-1])
            self.params["W"+str(idx)] = scale * np.random.rand(all_layer[idx-1],all_layer[idx])
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])

        
        return self.params