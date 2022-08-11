import numpy as np

def add_polynomial_features(x, power):
	"""Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power given in argument.
	Args:
	x: has to be an numpy.ndarray, a matrix of shape m * n.
	power: has to be an int, the power up to which the columns of matrix x are going to be raised.
	Returns:
	The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature values for all training examples.
	None if x is an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.size == 0:
		return None
	if not isinstance(power, int) or power <= 0:
		return None
	try:
		X = x
		for i in range(2, power + 1):
			X = np.concatenate((X, x ** i), axis=1)
		return X
	except:
		return None

