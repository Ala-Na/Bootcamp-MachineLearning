import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        print("x is not a numpy array of dimension m * 1")
        return
    elif not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] != x.shape[0]:
        print("y is not a numpy array of dimension m * 1")
        return
    elif not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return
    X = np.insert(x, 0, 1.0, axis=1)
    hyp = np.dot(X, theta)
    plt.plot(x, y, 'o')
    plt.plot(x, hyp)
    plt.show()


x = np.arange(1,6).reshape(-1, 1)
y = np.array([[3.74013816],[3.61473236],[4.57655287],[4.66793434],[5.95585554]])
theta1 = np.array([[4.5],[-0.2]])
plot(x, y, theta1)

theta2 = np.array([[-1.5],[2]])
plot(x, y, theta2)

theta3 = np.array([[3],[0.3]])
plot(x, y, theta3)
