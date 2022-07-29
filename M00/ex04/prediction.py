import numpy as np

def predict_(x, theta):
    '''Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    '''
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        print("x is not a non-empty numpy array of dimension m * 1")
        return None
    elif not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return None
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot(X, theta)
    except:
        print("something went wrong")
        return None

