import numpy as np

class MyLogisticRegression():
    """
    Description:
        My personnal logistic regression to classify things.
    """
	
    supported_penalities = ['l2', None] # We consider l2 penality only. One may wants to implement other penalities
	
    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
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
        assert penalty in self.supported_penalities
        assert isinstance(lambda_, float)
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty != None else 0
        if self.lambda_ < 0:
            raise ValueError("Lambda must be positive")

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
            m = y.shape[0]
            return ((-1 / m) * (y.T.dot(np.log(y_hat + eps)) + (one_vec - y).T.dot(np.log(one_vec - y_hat + eps))) + ((self.lambda_ / (2 * m)) * self.l2(self.theta))).item()
        except:
            return None

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        try:
            m = y.shape[0]
            X = np.insert(x, 0, 1.0, axis=1)
            y_hat = 1 / (1 + np.exp(-X @ self.theta)) # Calculus of linear regression prediction
            theta_cp = self.theta.copy()
            theta_cp[0][0] = 0
            return (1 / m) * (X.T.dot(y_hat - y) + self.lambda_ * theta_cp)
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

    def l2(self):
        try:
            theta_cp = self.theta.copy()
            theta_cp[0][0] = 0
            return np.sum(theta_cp ** 2)
        except:
            return None
