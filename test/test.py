import gpu
if gpu.Use_Gpu:
    import cupy as np
else:
    import numpy as np
from collections import OrderedDict
import copy

class Huh:
    def __init__ (self,ins,n):
        self.a = ins.s+n
