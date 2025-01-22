import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
import math
import copy

class He:
    def __init__(self):
        self.params = {}
    def weight_initialization(self, inp, layer, convlayer, out,batchnorm):
        conv_layer = copy.deepcopy(convlayer)
        if len(conv_layer) != 0:
            Ih = inp[1]
            Iw = inp[2]
            conv_layer.insert(0,inp)
            conv = [l for l in conv_layer if len(l) == 3]
                
            cc = 1
            for idx in range(1,len(conv_layer)):
            
                if len(conv_layer[idx]) == 3:
                    scale = np.sqrt(2.0/(conv[cc-1][0]*conv[cc][1]*conv[cc][1]))
                    self.params["F"+str(cc)] = scale * np.random.randn(conv[cc][0],conv[cc-1][0],conv[cc][1],conv[cc][1])
                    Ih = Ih +2* conv[cc][2] - conv[cc][1] + 1
                    Iw = Iw +2* conv[cc][2] - conv[cc][1] + 1
                    self.params["Cb"+str(cc)] = np.zeros((conv[cc][0]))
                    if batchnorm:
                        self.params["gamma"+str(cc)] = np.ones(conv[cc][0])
                        self.params["beta"+str(cc)] = np.zeros(conv[cc][0])
                        self.params["move_m"+str(cc)] = np.zeros(conv[cc][0])
                        self.params["move_v"+str(cc)] = np.ones(conv[cc][0])
                    cc += 1
        

                elif len(conv_layer[idx]) == 1:

                    vpad = hpad = (Ih - conv_layer[idx][0]) % 2
                    Ih = (Ih - conv_layer[idx][0] + vpad)//2 + 1
                    Iw = (Iw - conv_layer[idx][0] + hpad)//2 + 1
                        #パディングの操作
            all_layer = [Ih * Iw * conv[len(conv)-1][0]] + layer + [out]
            
        else:
            all_layer = [int(math.prod(inp))] + layer + [out]
            

        for idx in range(1,len(all_layer)):
            scale = np.sqrt(2.0/all_layer[idx-1])
            self.params["W"+str(idx)] = scale * np.random.randn(all_layer[idx-1],all_layer[idx])
            self.params["b"+str(idx)] = np.zeros(all_layer[idx])
        
        return self.params