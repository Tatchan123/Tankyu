import matplotlib.pyplot as pyp
import numpy as np
import openpyxl
import os


i = np.arange(10)
j = np.arange(10)*2


# sxy = np.cov(x,y)
# sx = np.var(x)
# sy = np.var(y)
# print(sxy)
# print(sxy[0][1] / np.sqrt(sx*sy))
# print(np.corrcoef(x,y))


sxy = np.cov(i,j)[0][1] # i=x , j=y
print(np.cov(i,j))
varx = np.var(i)
slope = sxy/(varx)


yinter = np.average(j) - 2*np.average(i)

print(sxy,varx)
print(slope,yinter)
