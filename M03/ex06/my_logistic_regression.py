import numpy as np

class MyLogisticRegression():

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        assert isinstance(alpha, float)
        assert isinstance(max_iter, int)
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.asarray(theta).reshape(-1, 1)

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return 1 / (1 + np.exp(-(X @ self.theta)))

    def loss_elem_(self, x):
        eps=1e-15
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        y_hat = self.predict_(x)
        return np.log(np.clip(y_hat, eps, 1 - eps)), np.log(np.clip(1 - y_hat, eps, 1 - eps))

    def loss_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        loss_elem = self.loss_elem_(x)
        return (float)(1 / y.shape[0]) * (((-y).T @ loss_elem[0]) - ((1 - y).T @ loss_elem[1])).item()

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return (1 / x.shape[0]) * (X.T @ (self.predict_(x) - y))

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        for i in range(0, self.max_iter):
            self.theta = self.theta - self.alpha * self.gradient_(x, y)