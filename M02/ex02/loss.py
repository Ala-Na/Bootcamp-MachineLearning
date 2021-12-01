import numpy as np

def loss_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.shape[1] != 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
        return None
    return np.sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0])
    

X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])

print(loss_(X, Y), '\n')
#Output : 2.1428571428571436

print(loss_(X, X))
#Output : 0