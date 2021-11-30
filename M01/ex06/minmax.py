import numpy as np

def minmax(x):
    if not isinstance(x, np.ndarray) and x.ndim != 2 and x.shape[1] != 1:
        return None
    normalize_minmax = lambda i: (i - np.amin(x)) / (np.amax(x) - np.amin(x))
    normalize_vect = np.vectorize(normalize_minmax)
    return normalize_vect(x)

X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
print(minmax(X), '\n')
#Output : array([0.58333333, 1., 0.33333333, 0.77777778, 0.91666667, 0.66666667, 0.])

Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(minmax(Y))
# Output: array([0.63636364, 1., 0.18181818, 0.72727273, 0.93939394, 0.6969697 , 0.])