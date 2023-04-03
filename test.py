import math
import numpy as np

a = 2*np.eye(3)
b = np.ones(shape=(3,3))
c = (-1)*np.eye(3)
print(np.dot(b,c))