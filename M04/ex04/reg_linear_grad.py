import numpy as np

def reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
	with two for-loop. The three arrays must have compatible shapes.
	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
	lambda_: has to be a float.
	Return:
	A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.
	None if y, x or theta or lambda_ is not of the expected type.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.size == 0 or y.shape[1] != 1:
		return None
	if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
		return None
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape != (x.shape[1] + 1, 1):
		return None
	if not isinstance(lambda_, float) or lambda_ < 0:
		return None
	try:
		m = y.shape[0]
		X = np.insert(x, 0, 1.0, axis=1)
		y_hat = np.dot(X, theta) # Calculus of linear regression prediction
		theta_cp = theta.copy()
		theta_cp[0][0] = 0
		costs = []
		for column in range(0, X.shape[1]):
			res = 0
			for row in range(0, X.shape[0]):
				res += ((y_hat[row][0] - y[row][0]) * X[row][column]).item()
			res = (1 / m) * (res + lambda_ * theta_cp[column][0])
			costs.append(res) 
		return np.array(costs).reshape(-1, 1)
	except:
		return None

def vec_reg_linear_grad(y, x, theta, lambda_):
	"""Computes the regularized linear gradient of three non-empty numpy.ndarray,
	without any for-loop. The three arrays must have compatible shapes.
	Args:
	y: has to be a numpy.ndarray, a vector of shape m * 1.
	x: has to be a numpy.ndarray, a matrix of dimesion m * n.
	theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
	lambda_: has to be a float.
	Return:
	A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
	None if y, x, or theta are empty numpy.ndarray.
	None if y, x or theta does not share compatibles shapes.
	None if y, x or theta or lambda_ is not of the expected type.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.size == 0 or y.shape[1] != 1:
		return None
	if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0 or x.shape[0] != y.shape[0] or x.shape[1] == 0:
		return None
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape != (x.shape[1] + 1, 1):
		return None
	if not isinstance(lambda_, float) or lambda_ < 0:
		return None
	try:
		m = y.shape[0]
		X = np.insert(x, 0, 1.0, axis=1)
		y_hat = np.dot(X, theta) # Calculus of linear regression prediction
		theta_cp = theta.copy()
		theta_cp[0][0] = 0
		return (1 / m) * (X.T.dot(y_hat - y) + lambda_ * theta_cp)
	except:
		return None
