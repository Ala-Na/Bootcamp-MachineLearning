import numpy as np

def gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(y, np.ndarray) or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1):
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.matmul((np.transpose(X) / x.shape[0]), (np.matmul(X, theta) - y))

x = np.array([[ -6, -7, -9],[ 13, -2, 14],[ -7, 14, -1],[ -8, -4, 6],[ -5, -9, 6],[ 1, -5, 11],[ 9, -11, 8]])
y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])

theta1 = np.array([[0],[ 3],[ 0.5],[ -6]])
print(gradient(x, y, theta1), '\n')
# Output: array([[ -33.71428571],[ -37.35714286],[ 183.14285714],[ -393.]])

theta2 = np.array([[0],[ 0],[ 0],[ 0]])
print(gradient(x, y, theta2), '\n')
# Output: array([[ -0.71428571],[ 0.85714286],[ 23.28571429],[ -26.42857143]])
