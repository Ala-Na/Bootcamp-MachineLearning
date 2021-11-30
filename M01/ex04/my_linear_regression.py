import numpy as np

class   MyLinearRegression():
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        assert isinstance(alpha, float)
        self.alpha = alpha
        assert isinstance(max_iter, int)
        self.max_iter= max_iter
        self.thetas = np.asarray(thetas).reshape((2, 1))

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot(X, self.thetas)

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        X = np.insert(x, 0, 1.0, axis = 1)
        X_T = X.transpose()
        y_hat = np.dot(X, self.thetas)
        return np.dot(X_T, y_hat - y) / x.shape[0]

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        for i in range(0, self.max_iter):
            curr_gradient = self.gradient_(x, y)
            theta_zero = (float)(self.thetas[0][0] - self.alpha * curr_gradient[0][0])
            theta_one = (float)(self.thetas[1][0] - self.alpha * curr_gradient[1][0])
            new_thetas = np.asarray([theta_zero, theta_one]).reshape((2, 1))
            self.thetas = new_thetas

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1:
            return None        
        J_elem = []
        for yi, yi_hat in zip(y, y_hat):
            J_elem.append([(yi_hat[0] - yi[0]) ** 2])
        return np.array(J_elem)

    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1:
            return None
        J_elem = self.loss_elem_(y, y_hat)
        J_value = float(1/(2*y.shape[0]) * np.sum(J_elem))
        return J_value

    def mse_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
            return None
        mse = ((y_hat - y) ** 2).mean(axis=None)
        return float(mse)