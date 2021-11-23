import numpy as np

def loss_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
        return None
    res = 0
    for yi, yi_hat in zip(y, y_hat):
        res += (yi_hat[0] - yi[0]) ** 2
    return res / ( 2 * y.shape[0])

X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(loss_(X, Y))
print(loss_(X, X))