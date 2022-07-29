import numpy as np
from tools import add_intercept

print(add_intercept.__doc__)

x = np.arange(1,6).reshape((5,1))
x = add_intercept(x)
print(x)

x = np.arange(1,10).reshape((3,3))
x = add_intercept(x)
print(x)

print(add_intercept(np.array([[]])))
