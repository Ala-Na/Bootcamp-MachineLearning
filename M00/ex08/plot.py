import numpy as np
import matplotlib.pyplot as plt

def predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        print("x is not a numpy array of dimension m * 1")
        return None
    elif not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.dot(X, theta)

def loss_elem_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1:
        return None        
    J_elem = []
    for yi, yi_hat in zip(y, y_hat):
        J_elem.append([(yi_hat[0] - yi[0]) ** 2])
    return np.array(J_elem)

def loss_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
        return None
    res = 0
    for yi, yi_hat in zip(y, y_hat):
        res += (yi_hat[0] - yi[0]) ** 2
    return res / ( 2 * y.shape[0])

def plot_with_loss(x, y, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        print("x is not a numpy array of dimension m * 1")
        return
    elif not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] != x.shape[0]:
        print("y is not a numpy array of dimension m * 1")
        return
    elif not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return
    y_hat = predict_(x, theta)
    cost = loss_(y, y_hat)
    plt.title("Cost = {:.06f}".format(cost))
    plt.plot(x, y, 'o')
    plt.plot(x, y_hat)
    for i in range(x.shape[0]):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], 'r--')
    plt.show()

x = np.arange(1,6).reshape(-1, 1)
y = np.array([[11.52434424],[10.62589482],[13.14755699],[18.60682298],[14.14329568]])
theta1= np.array([[18],[-1]])
plot_with_loss(x, y, theta1)

theta2 = np.array([[14],[0]])
plot_with_loss(x, y, theta2)

theta3 = np.array([[12],[0.8]])
plot_with_loss(x, y, theta3)