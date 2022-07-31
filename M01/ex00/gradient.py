import numpy as np

# NO FOR LOOP

def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    y: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
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
        y_hat = np.dot(X, theta)
        j0 = 0
        j1 = 0
        for i in range(y.shape[0]):
            j0 += y_hat[i] - y[i]
            j1 += (y_hat[i] - y[i]) * x[i]
        j0 /= x.shape[0]
        j1 /= x.shape[0]
        return np.asarray([j0, j1]).reshape((2, 1))
    except:
        print("Something went wrong.")
        return None
