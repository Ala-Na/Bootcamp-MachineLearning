import numpy as np


class   MyLinearRegression():

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        assert isinstance(alpha, float)
        assert isinstance(max_iter, int)
        if isinstance(thetas, np.ndarray):
            assert np.issubdtype(thetas.dtype, np.number)
            self.thetas = thetas
        else:
            try:
                self.thetas = np.asarray(thetas).reshape((len(thetas), 1))
                assert np.issubdtype(self.thetas.dtype, np.number)
            except:
                raise ValueError("Thetas not valid")
        self.alpha = alpha
        self.max_iter= max_iter


    def predict_(self, x):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or (x.shape[1] + 1) != self.thetas.shape[0]:
            return None
        try:
            X = np.insert(x, 0, 1.0, axis=1)
            return np.dot(X, self.thetas)
        except:
            return None

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or (x.shape[1] + 1) != self.thetas.shape[0]:
            return None
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.shape != (x.shape[0], 1):
            return None
        try:
            X = np.insert(x, 0, 1.0, axis=1)
            return np.dot((np.transpose(X) / x.shape[0]), (np.dot(X, self.thetas) - y))
        except:
            return None

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or (x.shape[1] + 1) != self.thetas.shape[0]:
            return None
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.shape != (x.shape[0], 1):
            return None
        try:
            for i in range(0, self.max_iter):
                curr_gradient = self.gradient_(x, y)
                new_theta = []
                for val in range(0, self.thetas.shape[0]):
                    new_theta.append((float)(self.thetas[val][0] - self.alpha * curr_gradient[val][0]))
                self.thetas = np.asarray(new_theta).reshape(self.thetas.shape)
            return self.thetas
        except:
            return None

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape[0] == 0 or y_hat.shape[1] != 1:
            return None        
        try:
            return (y_hat - y) ** 2
        except:
            return None

    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[0] == 0 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
            return None
        try:
            return np.sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0])
        except:
            return None
