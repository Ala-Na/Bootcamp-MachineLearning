import numpy as np

def predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        print("x is not a numpy array of dimension m * 1")
        return None
    elif not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.dot(X, theta)

x = np.arange(1,6).reshape(-1, 1)

theta1 = np.array([[5],[0]])
print(predict_(x, theta1))

theta2 = np.array([[0],[1]])
print(predict_(x, theta2))

theta3 = np.array([[5],[3]])
print(predict_(x, theta3))

theta4 = np.array([[-3],[1]])
print(predict_(x, theta4))
