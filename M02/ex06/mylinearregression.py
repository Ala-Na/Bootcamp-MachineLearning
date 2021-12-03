import numpy as np

class   MyLinearRegression():

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        assert isinstance(alpha, float)
        self.alpha = alpha
        assert isinstance(max_iter, int)
        self.max_iter= max_iter
        self.thetas = np.asarray(thetas).reshape((len(thetas), 1))

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or (x.shape[1] + 1) != self.thetas.shape[0]:
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot(X, self.thetas)

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.shape != (x.shape[0], 1):
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot((np.transpose(X) / x.shape[0]), (np.dot(X, self.thetas) - y))


    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        for i in range(0, self.max_iter):
            curr_gradient = self.gradient_(x, y)
            new_theta = []
            for val in range(0, self.thetas.shape[0]):
                new_theta.append((float)(self.thetas[val][0] - self.alpha * curr_gradient[val][0]))
            self.thetas = np.asarray(new_theta).reshape(self.thetas.shape)
        return self.thetas

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1:
            return None        
        return (y_hat - y) ** 2

    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.shape[1] != 1:
            return None
        if not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
            return None
        return np.sum((y_hat - y) * (y_hat - y)) / (2 * y.shape[0])

    def mse_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
            return None
        elif not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
            return None
        mse = ((y_hat - y) ** 2).mean(axis=None)
        return float(mse)