import numpy as np
import math

def mse_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
        return None
    mse = ((y_hat - y) ** 2).mean(axis=None)
    return float(mse)

def rmse_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
        return None
    rmse = math.sqrt(mse_(y, y_hat))
    return float(rmse)

def mae_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
        return None
    mae = (np.absolute(y_hat - y)).mean(axis=None)
    return float(mae)

def r2score_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
        return None
    y_bar = y.mean(axis=None)
    r2score = 1 - (np.sum((y_hat - y) ** 2) / np.sum((y - y_bar) ** 2))
    return float(r2score)

x = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(mse_(x,y))
print(rmse_(x,y))
print(mae_(x,y))
print(r2score_(x, y))