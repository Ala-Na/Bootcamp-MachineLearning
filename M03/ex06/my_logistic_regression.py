import numpy as np

class MyLogisticRegression():

    def __init__(self, theta, alpha=0.0001, max_iter=1000):
        assert isinstance(alpha, float)
        assert isinstance(max_iter, int)
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = np.asarray(theta).reshape(-1, 1)

    def predict_(self, x):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return 1 / (1 + np.exp(np.dot(-X, self.theta)))

    def loss_elem_(self, x):
        eps=1e-15
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        y_hat = self.predict_(x)
        return np.log(eps + y_hat), np.log(eps + 1 - y_hat)

    def loss_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            return None
        loss_elem = self.loss_elem_(x)
        return - (np.sum(y * loss_elem[0] + (1 - y) * loss_elem[1]) / y.shape[0])

    def gradient_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != self.theta.shape[0] - 1:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[0] != x.shape[0] or y.shape[1] != 1:
            return None
        X = np.insert(x, 0, 1.0, axis=1)
        return (np.dot(X.transpose(), (self.predict_(x) - y))) / x.shape[0]

    def fit_(self, x, y):
        if not isinstance(x, np.ndarray) or x.ndim != 2:
            return None
        if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape != (x.shape[0], 1):
            return None
        for i in range(0, self.max_iter):
            curr_gradient = self.gradient_(x, y)
            new_theta = []
            #print(curr_gradient)
            for val in range(0, self.theta.shape[0]):
                #print("Theta:", self.theta[val, 0])
                #print("Grad: ", curr_gradient[val, 0])
                new_theta.append((float)(self.theta[val, 0] - self.alpha * curr_gradient[val, 0]))
            self.theta = np.asarray(new_theta).reshape(self.theta.shape)
        return self.theta