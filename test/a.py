import matplotlib.pyplot as pyp
import numpy as np
import openpyxl
import os

a = np.arange(15).reshape(3,5)
b = np.array([0,0])

o = np.insert(a,[1,3],b,axis=0)
print(o)