import numpy as np

def simple_predict(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        print("x is not a numpy array of dimension m * 1")
        return None
    elif not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return None
    res = np.zeros((x.shape[0], 1))
    for ind in range(x.shape[0]):
        value = theta[0][0] + theta[1][0] * x[ind][0]
        res[ind][0] = value
    return res

