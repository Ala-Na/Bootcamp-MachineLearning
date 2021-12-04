import numpy as np

def united_shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def data_spliter(x, y, proportion):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(y, np.ndarray) or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(proportion, float) or proportion > 1 or proportion < 0:
        return None
    ind_split = (int)(x.shape[0] * proportion)
    x, y = united_shuffle(x, y)
    return (x[:ind_split, :], x[ind_split:, :], y[:ind_split, :], y[ind_split:, :])
    



x1 = np.array([[1],[ 42],[ 300],[ 10],[ 59]])
y = np.array([[0],[1],[0],[1],[0]])

# Example 0:
print(data_spliter(x1, y, 0.8))
# example output: (array([[ 10],[ 42],[ 1],[ 300]]), array([[59]]), array([[1],[ 1],[ 0],[ 0]]), array([[0]]))

# Example 1:
print(data_spliter(x1, y, 0.5))
# example output: (array([[42],[ 10]]), array([[ 59],[ 300],[ 1]]), array([[1],[ 1]]), array([[0],[ 0],[ 0]]))

x2 = np.array([ [ 1, 42],[300, 10],[ 59, 1],[300, 59],[ 10, 42]])
y = np.array([[0],[1],[0],[1],[0]])

# Example 2:
print(data_spliter(x2, y, 0.8))
# example output:(array([[ 10, 42],[ 59, 1],[ 1, 42],[300, 10]]), array([[300, 59]]), array([[0],[ 0],[ 0],[ 1]]),array([[1]]))

# Example 3:
print(data_spliter(x2, y, 0.5))
# example output:(array([[300, 10],[ 1, 42]]),array([[ 10, 42],[300, 59],[ 59, 1]]),array([[1],[ 0]]),array([[0],[ 1],[ 0]]))