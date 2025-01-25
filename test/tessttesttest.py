import numpy as np

x = np.random.rand(10)
z = np.random.randn(10)
print(x)
zipped = zip(x,z)
zx,zz = zip(*zipped)
y = [np.array(i).tolist() for i in zx]
print(y)