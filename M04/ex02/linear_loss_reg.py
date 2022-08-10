import numpy as np

def l2(theta):
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape[1] != 1:
		return None
	try:
		theta[0][0] = 0
		return np.sum(theta ** 2)
	except:
		return None


def reg_loss_(y, y_hat, theta, lambda_):
	"""Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop. The two arrays must have the same dimensions.
	Args:
	y: has to be an numpy.ndarray, a vector of shape m * 1.
	y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	lambda_: has to be a float.
	Returns:
	The regularized loss as a float.
	None if y, y_hat, or theta are empty numpy.ndarray.
	None if y and y_hat do not share the same shapes.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.size == 0 or y.shape[1] != 1:
		return None
	if not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape != y.shape:
		return None
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape[1] != 1:
		return None
	if not isinstance(lambda_, float) or lambda_ < 0:
		return None
	try:
		return (1 / (2 * y.shape[0])) * (np.sum((y_hat - y) ** 2) + lambda_ * l2(theta))
	except:
		return None
