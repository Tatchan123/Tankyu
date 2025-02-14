import cupy as np
from collections import OrderedDict
import copy

a = np.random.randint(0,100,10)
order = np.argsort(a)
print(a[order])

