import numpy as np

def logistic_predict_(x, theta):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.shape != (x.shape[1] + 1, 1):
        return None
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        return 1 / (1 + np.exp(-X @ theta))
    except:
        return None

def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: has to be a float, epsilon (default=1e-15)
    Returns:
    The logistic loss value as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number)or y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != 1:
        return None
    if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
        return None
    if not isinstance(eps, float):
        return None
    try:
        return (float)(1 / y.shape[0])*(((-y).T @ np.log(np.clip(y_hat, eps, 1 - eps))) - ((1 - y).T @ np.log(np.clip(1 - y_hat, eps, 1 - eps)))).item()
    except:
        return None

# This function is called Cross-Entropy loss, or logistic loss.

# The logarithmic function isn’t defined in 0. This means that if
# y(i) = 0 you will get an error when you try to compute log(y(i)). The
# purpose of the eps argument is to avoid log(0) errors. It is a very
# small residual value we add to y
