import numpy as np


class   MyLinearRegression():

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        assert isinstance(alpha, float)
        self.alpha = alpha
        assert isinstance(max_iter, int)
        self.max_iter= max_iter
        self.thetas = np.asarray(thetas).reshape((2, 1))

    def predict_(x, theta):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(theta, np.ndarray) or theta.shape[0] != x.shape[1] + 1:
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot(X, theta)

    def gradient_(x, y, theta):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.shape != (x.shape[0], 1):
            return None
        if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1):
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return np.matmul((np.transpose(X) / x.shape[0]), (np.matmul(X, theta) - y))

    def fit_(x, y, theta, alpha, max_iter):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1):
            return None
        if not isinstance(alpha, float) or not isinstance(max_iter, int):
            return None
        for i in range(0, max_iter):
            curr_gradient = gradient_(x, y, theta)
            new_theta = []
            for val in range(0, theta.shape[0]):
                new_theta.append((float)(theta[val][0] - alpha * curr_gradient[val][0]))
            theta = np.asarray(new_theta).reshape(theta.shape)
        return theta

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1:
            return None        
        J_elem = []
        for yi, yi_hat in zip(y, y_hat):
            J_elem.append([(yi_hat[0] - yi[0]) ** 2])
        return np.array(J_elem)

    def loss_(y, y_hat):
        if not isinstance(y, np.ndarray) or y.shape[1] != 1:
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
            return None
        return np.sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0])