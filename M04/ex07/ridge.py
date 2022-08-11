from mylinearregression import MyLinearRegression as MyLR
import numpy as np

class MyRidge(MyLR):
	"""
	Description:
		My personnal ridge regression class to fit like a boss.
	"""
	
	def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
		assert isinstance(lambda_, float) and lambda_ >= 0
		super(MyRidge, self).__init__(thetas, alpha, max_iter)
		self.lambda_ = lambda_

	def get_params_(self):
		return {'thetas': self.theta, 'alpha': self.alpha, 'max_iter': self.max_iter, 'lambda_': self.lambda_}

	def set_params_(self, thetas=None, alpha=None, max_iter=None, lambda_=None):
		if thetas is not None and isinstance(thetas, np.ndarray) and np.issubdtype(thetas.dtype, np.number) and thetas.size != 0:
			self.theta = thetas
		elif thetas is not None:
			self.theta = np.asarray(thetas).reshape(-1, 1)
			if not np.issubdtype(self.theta.dtype, np.number) or self.theta.size == 0:
				raise ValueError("Thetas not valid")
		if alpha is not None and isinstance(alpha, float):
			self.alpha = alpha
		elif alpha is not None:
			raise ValueError("Alpha not valid")
		if max_iter is not None and isinstance(max_iter, int):
			self.max_iter = max_iter
		elif max_iter is not None:
			raise ValueError("Max_iter not valid")
		if lambda_ is not None and isinstance(lambda_, float) and lambda_ >= 0:
			self.lambda_ = lambda_
		elif lambda_ is not None:
			raise ValueError("Lambda not valid")
		return self.get_params_()

	def loss_(self, y, y_hat):
		if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.size == 0 or y.shape[1] != 1:
			return None
		if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
			return None
		try:
			theta_cp = self.theta.copy()
			theta_cp[0][0] = 0
			l2 = np.sum(theta_cp ** 2)
			return (1 / (2 * y.shape[0])) * (np.sum((y_hat - y) ** 2) + self.lambda_ * l2)
		except:
			return None

	def loss_elem(self, y, y_hat):
		if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.size == 0 or y.shape[1] != 1:
			return None
		if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
			return None
		try:
			theta_cp = self.theta.copy()
			theta_cp[0][0] = 0
			l2 = np.sum(theta_cp ** 2)
			return ((y_hat - y) ** 2) + self.lambda_ * l2
		except:
			return None		

	def gradient_(self, x, y):
		if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.size == 0 or y.shape[1] != 1:
			return None
		if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0 or x.shape[0] != y.shape[0]:
			return None
		if not (x.shape[1] + 1) != self.theta.shape[0] and not x.shape[1] != self.theta.shape[0]:
			return None
		try:
			m = y.shape[0]
			if x.shape[1] != self.theta.shape[0]:
				X = np.insert(x, 0, 1.0, axis=1)
			else:
				X = x.copy()
			y_hat = np.dot(X, self.theta)
			theta_cp = self.theta.copy()
			theta_cp[0][0] = 0
			return (1 / m) * (X.T.dot(y_hat - y) + (self.lambda_ * theta_cp))
		except:
			return None


	def fit_(self, x, y):
		if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or (x.shape[1] + 1) != self.theta.shape[0]:
			return None
		if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.shape != (x.shape[0], 1):
			return None
		try:
			for i in range(0, self.max_iter):
				curr_gradient = self.gradient_(x, y)
				self.theta = self.theta - (self.alpha * curr_gradient)
			return self.theta
		except:
			return None
