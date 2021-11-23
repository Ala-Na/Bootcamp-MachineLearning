import numpy as np

def add_intercept(x):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        print("x is not a numpy array of dimension m * 1")
        return None
    x = np.insert(x, 0, 1.0, axis=1)
    return x

x = np.arange(1,6).reshape((5,1))
x = add_intercept(x)
print(x)

x = np.arange(1,10).reshape((3,3))
x = add_intercept(x)
print(x)