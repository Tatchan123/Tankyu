import cupy as np
from collections import OrderedDict
import copy

rm = np.arange(5)
co = np.arange(5)

g = zip(np.meshgrid(rm,co))
print(*g)