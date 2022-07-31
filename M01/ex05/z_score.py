import numpy as np
import math

def mean_(x):
    try:
        return np.sum(x) / x.shape[0]
    except:
        return None

def var_(x):
    try:
        mean = mean_(x)
        fun = lambda i : (i - mean) ** 2
        fun_vect = np.vectorize(fun)
        return np.sum(fun_vect(x) / x.shape[0])
    except:
        return None

def std_(x):
    try:
        var = var_(x)
        return math.sqrt(var) 
    except:
        return None

def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or \
        not (x.ndim == 1 or (x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1))):
        return None
    if x.shape[0] == 0 or (x.ndim == 2 and x.shape[1] == 0):
        return None
    try:
        mean = mean_(x)
        std = std_(x)
        normalize = lambda i: (i - mean) / std
        normalize_vect = np.vectorize(normalize)
        return (normalize_vect(x).reshape(1, -1))[0]
    except:
        return None

