import numpy as np

def minmax(x):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or \
        not (x.ndim == 1 or (x.ndim == 2 and (x.shape[0] == 1 or x.shape[1] == 1))):
        return None
    if x.shape[0] == 0 or (x.ndim == 2 and x.shape[1] == 0):
        return None
    normalize_minmax = lambda i: (i - np.amin(x)) / (np.amax(x) - np.amin(x))
    normalize_vect = np.vectorize(normalize_minmax)
    return normalize_vect(x).reshape(1, -1)[0]
