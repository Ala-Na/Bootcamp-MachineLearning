import numpy as np

def iterative_l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
	Args:
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape[1] != 1:
		return None
	try:
		res = 0
		for i in range(theta.shape[0]):
			res += theta[i][0] ** 2
		return res
	except:
		return None

def l2(theta):
	"""Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
	Args:
	theta: has to be a numpy.ndarray, a vector of shape n * 1.
	Returns:
	The L2 regularization as a float.
	None if theta in an empty numpy.ndarray.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.size == 0 or theta.shape[1] != 1:
		return None
	try:
		return np.sum(theta ** 2)
	except:
		return None
