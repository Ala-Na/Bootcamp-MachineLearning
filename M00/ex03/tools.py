import numpy as np

def add_intercept(x):
    '''Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    '''
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        print("x is not a non-empty numpy array of dimension m * n")
        return None
    try:
        x = np.insert(x, 0, 1.0, axis=1)
        return x
    except:
        print("Something went wrong")
        return None
