import matplotlib.pyplot as pyp
import numpy as np
import openpyxl
import os

ma = np.empty((5,5))
x = np.arange(5,11)
y = np.arange(5)+1
rm = np.arange(5)
co = np.arange(5)
ma = np.tile(x,(5,1)).T * y

mu = np.mean(ma,1)


sab = np.cov(ma)
cor = np.corrcoef(ma)
a = np.zeros_like(ma,dtype=float)
b = np.zeros_like(ma,dtype=float)

for j in co:
    a[:,j] = sab[:,j] / (sab[j,j] + 1e-8)
    b[:,j] = mu - a[:,j] * mu[j]


