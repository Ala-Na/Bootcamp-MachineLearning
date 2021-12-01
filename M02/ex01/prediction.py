import numpy as np

def predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape[0] != x.shape[1] + 1:
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.dot(X, theta)

x = np.arange(1,13)
x = np.reshape(x, (4,3))

theta1 = np.array([[5],[ 0],[ 0],[ 0]])
print(predict_(x, theta1), '\n')
# Ouput: array([[5.],[ 5.],[ 5.],[ 5.]])

theta2 = np.array([[0],[ 1],[ 0],[ 0]])
print(predict_(x, theta2), '\n')
# Output: array([[ 1.],[ 4.],[ 7.],[ 10.]])

theta3 = np.array([[-1.5],[ 0.6],[ 2.3],[ 1.98]])
print(predict_(x, theta3), '\n')
# Output: array([[ 9.64],[ 24.28],[ 38.92],[ 53.56]])

theta4 = np.array([[-3],[ 1],[ 2],[ 3.5]])
print(predict_(x, theta4))
# Output: array([[12.5],[ 32. ],[ 51.5],[ 71. ]])