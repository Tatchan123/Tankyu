import matplotlib.pyplot as pyp
import numpy as np
import openpyxl
import os

x = np.arange(5,11)
y = np.arange(5)+1
rm,co = np.meshgrid(np.arange(6),np.arange(6))
ma = np.tile(x,(5,1)).T * y

mu = np.mean(ma,1)


sab = np.cov(ma)
cor = np.corrcoef(ma)
a = np.zeros_like(sab,dtype=float)
b = np.zeros_like(sab,dtype=float)

for j in range(sab.shape[0]):
    a[j] = sab[j] / (sab[j,j] + 1e-8)
    b[j] = mu - a[j] * mu[j]
print(ma)
print(a)
print(ma[2]- (a[0,2]* ma[0]+ b[0,2]))
