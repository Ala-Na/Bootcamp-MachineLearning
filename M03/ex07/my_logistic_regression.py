import numpy as np

class MyLogisticRegression():

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        assert isinstance(alpha, float)
        assert isinstance(max_iter, int)
        if isinstance(theta, np.ndarray):
            assert np.issubdtype(theta.dtype, np.number)
            self.theta = theta
        else:
            try:
                self.theta = np.asarray(theta).reshape((len(theta), 1))
                assert np.issubdtype(self.theta.dtype, np.number)
            except:
                raise ValueError("Thetas not valid")
        self.alpha = alpha
        self.max_iter= max_iter

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        try:
            X = np.insert(x, 0, 1.0, axis=1)
            return 1 / (1 + np.exp(-X @ self.theta))
        except:
            return None

    def loss_elem_(self, y, y_hat):
        eps=1e-15
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != 1:
            return None
        if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
            return None
        try:
            one_vec = np.ones((1, y.shape[1]))
            return (y * np.log(np.clip(y_hat, eps, 1 - eps))) + ((one_vec - y) * np.log(np.clip(one_vec - y_hat, eps, 1 - eps)))
        except:
            return None

    def loss_(self, y, y_hat):
        eps=1e-15
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number)or y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != 1:
            return None
        if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
            return None
        if not isinstance(eps, float):
            return None
        try:
            one_vec = np.ones((1, y.shape[1]))
            return (float)(1 / y.shape[0])*(((-y).T @ np.log(np.clip(y_hat, eps, 1 - eps))) - ((one_vec - y).T @ np.log(np.clip(one_vec - y_hat, eps, 1 - eps)))).item()
        except:
            return None

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        try:
            X = np.insert(x, 0, 1.0, axis=1)
            return (1 / x.shape[0]) * (X.T @ (self.predict_(x) - y))
        except:
            return None

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        try:
            for i in range(0, self.max_iter):
                self.theta = self.theta - self.alpha * self.gradient_(x, y)
        except:
            return None
