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

#Same thoughts as ex02
def vec_log_loss_(y, y_hat, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
    eps: epsilon (default=1e-15)
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
        one_vec = np.ones((1, y.shape[1]))
        return (float)(1 / y.shape[0])*(((-y).T @ np.log(np.clip(y_hat, eps, 1 - eps))) - ((one_vec - y).T @ np.log(np.clip(one_vec - y_hat, eps, 1 - eps)))).item()
        #return - (np.sum(y * np.log(eps + y_hat) + (one_vec - y) * np.log(eps + one_vec - y_hat)) / y.shape[0])
    except:
        return None
