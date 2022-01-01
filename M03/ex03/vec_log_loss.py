import numpy as np

def logistic_predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1:
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return 1 / (1 + np.exp(np.dot(-X, theta)))

def vec_log_loss_(y, y_hat, eps=1e-15):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1 or y_hat.shape[0] != y.shape[0]:
        return None
    if not isinstance(eps, float):
        return None
    
