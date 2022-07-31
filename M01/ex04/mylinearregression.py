import numpy as np

class   MyLinearRegression():

    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        assert isinstance(alpha, float)
        assert isinstance(max_iter, int)
        assert (isinstance(thetas, np.ndarray) and np.issubdtype(thetas.dtype, np.number) and thetas.shape == (2, 1))
        self.alpha = alpha
        self.max_iter= max_iter
        self.thetas = thetas

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
            return None
        try:
            X = np.insert(x, 0, 1.0, axis=1)
            return np.dot(X, self.thetas)
        except:
            return None

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
            return None
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
            return None
        try:
            X = np.insert(x, 0, 1.0, axis = 1)
            X_T = X.transpose()
            y_hat = np.dot(X, self.thetas)
            return np.dot(X_T, y_hat - y) / x.shape[0]
        except:
            return None

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
            return None
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
            return None
        try:
            for i in range(0, self.max_iter):
                curr_gradient = self.gradient_(x, y)
                theta_zero = (float)(self.thetas[0][0] - self.alpha * curr_gradient[0][0])
                theta_one = (float)(self.thetas[1][0] - self.alpha * curr_gradient[1][0])
                new_theta = np.asarray([theta_zero, theta_one]).reshape((2, 1))
                if np.array_equal(new_theta, self.thetas):
                    break
                self.thetas = new_theta
        except:
            return None

    def loss_elem_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
            return None
        if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape[1] != 1 or y_hat.shape[0] == 0:
            return None
        try:        
            J_elem = []
            for yi, yi_hat in zip(y, y_hat):
                J_elem.append([(yi_hat[0] - yi[0]) ** 2])
            return np.array(J_elem)
        except:
            return None

    def loss_(self, y, y_hat):
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
            return None
        if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape[1] != 1 or y_hat.shape[0] == 0:
            return None  
        try:
            J_elem = self.loss_elem_(y, y_hat)
            J_value = float(1/(2*y.shape[0]) * np.sum(J_elem))
            return J_value
        except:
            return None

    def mse_(y, y_hat):
            if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
                return None
            if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape[1] != 1 or y_hat.shape[0] == 0:
                return None
            mse = ((y_hat - y) ** 2).mean(axis=None)
            return float(mse)
