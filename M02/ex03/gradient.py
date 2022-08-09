import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible dimensions.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.shape != (x.shape[1] + 1, 1):
        return None
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        return np.matmul((np.transpose(X) / x.shape[0]), (np.matmul(X, theta) - y))
    except:
        return None
