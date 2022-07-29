import numpy as np

def loss_(y, y_hat):
    '''Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    '''
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.shape != y.shape:
        return None
    try:
        return np.sum(((y - y_hat) ** 2) / (2 * y.shape[0]))
    except:
        print("Something went wrong")
        return None
