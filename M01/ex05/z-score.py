import numpy as np
import math

def mean_(x):
    if not isinstance(x, np.ndarray) and x.ndim != 2 and x.shape[1] != 1:
        return None
    return np.sum(x) / x.shape[0]

def var_(x):
    if not isinstance(x, np.ndarray) and x.ndim != 2 and x.shape[1] != 1:
        return None
    mean = mean_(x)
    fun = lambda i : (i - mean) ** 2
    fun_vect = np.vectorize(fun)
    return np.sum(fun_vect(x) / x.shape[0])

def std_(x):
    var = var_(x)
    return math.sqrt(var) 

def zscore(x):
    if not isinstance(x, np.ndarray) and x.ndim != 2 and x.shape[1] != 1:
        return None
    mean = mean_(x)
    std = std_(x)
    normalize = lambda i: (i - mean) / std
    normalize_vect = np.vectorize(normalize)
    return normalize_vect(x)


X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
print(zscore(X), '\n')
#Output : array([-0.08620324, 1.2068453 , -0.86203236, 0.51721942, 0.94823559, 0.17240647, -1.89647119])
Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(zscore(Y))
#Output: array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659, 0.28795027, -1.72770165])
