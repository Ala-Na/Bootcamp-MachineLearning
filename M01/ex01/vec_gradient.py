import numpy as np

#NO FOR LOOP

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.shape != (2, 1):
        return None
    try:
        X = np.insert(x, 0, 1.0, axis = 1)
        X_T = X.transpose()
        y_hat = np.dot(X, theta)
        return np.dot(X_T, y_hat - y) / x.shape[0]
    except:
        print("Something went wrong")
        return None
