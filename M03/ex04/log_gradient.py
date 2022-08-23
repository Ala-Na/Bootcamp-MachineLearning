import math
import numpy as np

def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number )or theta.ndim != 2 or theta.shape != (x.shape[1] + 1, 1):
        return None
    # try:
    X = np.insert(x, 0, 1.0, axis=1)
    grad = 0
    y_hat = 1 / (1 + np.exp(-X @ theta))
    for i in range(x.shape[0]):
        grad += X[i] * (y_hat[i] - y[i])
    return (grad / y.shape[0]).reshape(-1, 1)
    # except:
    #     return None
