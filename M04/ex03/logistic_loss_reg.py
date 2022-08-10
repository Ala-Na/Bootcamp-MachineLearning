import numpy as np

def l2(theta):
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape[1] != 1:
		return None
	try:
		theta[0][0] = 0
		return np.sum(theta ** 2)
	except:
		return None


def reg_log_loss_(y, y_hat, theta, lambda_):
	"""Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for loop. The two arrays must have the same shapes.
	Args:
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	lambda_: has to be a float.
	Returns:
	The regularized loss as a float.
	None if y, y_hat, or theta is empty numpy.ndarray.
	None if y and y_hat do not share the same shapes.
	Raises:
	This function should not raise any Exception.
	"""
	eps = 1e-15
	if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.size == 0 or y.shape[1] != 1:
		return None
	if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
		return None
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape[1] != 1:
		return None
	if not isinstance(lambda_, float) or lambda_ < 0:
		return None
	try:
		one_vec = np.ones((1, y.shape[1]))
		m = y.shape[0]
		return ((-1 / m) * (y.T.dot(np.log(y_hat + eps)) + (one_vec - y).T.dot(np.log(one_vec - y_hat + eps))) + ((lambda_ / (2 * m)) * l2(theta))).item()
		# Note : probably better to use np.clip(y_hat, eps, 1 - eps) inside np.log(), but subject result obtained with previous formula
	except:
		return None	
