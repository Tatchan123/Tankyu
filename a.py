import matplotlib.pyplot as pyp
import numpy as np
a = np.array([1,2,3,4,5])
b = a*3
ax,fig = pyp.subplots()
fig.plot(a,b)
pyp.show()
