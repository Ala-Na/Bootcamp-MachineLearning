import numpy as np

# Vandermonde matrix (x | x2 | x3 | ... | xn)

def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
    The matrix of polynomial features as a numpy.array, of dimension m * n,
    containing the polynomial feature values for all training examples.
    None if x is an empty numpy.array.
    None if x or power is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        return None
    if not isinstance(power, int) or power <= 0:
        return None
    try:
        X = x
        for i in range(1, power):
            X = np.append(X, ((X[:,0] ** (i + 1)).reshape(-1, 1)), axis=1)
        return X
    except:
        return None
