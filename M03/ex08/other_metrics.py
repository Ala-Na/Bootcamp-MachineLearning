# Those metrics functions are available in sklearn.metrics

import numpy as np

def accuracy_score_(y, y_hat):
	"""
	Compute the accuracy score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
	Returns:
		The accuracy score as a float.
		None on any error.
	Raises:
		This function should not raise any Exception.
	"""
	if not isinstance(y, np.ndarray) or y.size == 0:
		return None
	if not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
		return None
	try:
		return np.sum(y == y_hat) / y.size
	except:
		return None


def precision_score_(y, y_hat, pos_label=1):
	"""
	Compute the precision score.
	Args:
		y:a numpy.ndarray for the correct labels
		y_hat:a numpy.ndarray for the predicted labels
		pos_label: str or int, the class on which to report the precision_score (default=1)
	Return:
	The precision score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(y, np.ndarray) or y.size == 0:
		return None
	if not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
		return None
	if not isinstance(pos_label, int) and not isinstance(pos_label, str):
		return None
	try:
		truePos = np.sum((y == pos_label) & (y_hat == pos_label))
		falsePos = np.sum((y != pos_label) & (y_hat == pos_label))
		return truePos / (truePos + falsePos)
	except:
		return None

def recall_score_(y, y_hat, pos_label=1):
	"""
	Compute the recall score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Return:
	The recall score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(y, np.ndarray) or y.size == 0:
		return None
	if not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
		return None
	if not isinstance(pos_label, int) and not isinstance(pos_label, str):
		return None
	try:
		truePos = np.sum((y == pos_label) & (y_hat == pos_label))
		falseNeg = np.sum((y == pos_label) & (y_hat != pos_label))
		return truePos / (truePos + falseNeg)
	except:
		return None
	

def f1_score_(y, y_hat, pos_label=1):
	"""
	Compute the f1 score.
	Args:
	y:a numpy.ndarray for the correct labels
	y_hat:a numpy.ndarray for the predicted labels
	pos_label: str or int, the class on which to report the precision_score (default=1)
	Returns:
	The f1 score as a float.
	None on any error.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(y, np.ndarray) or y.size == 0:
		return None
	if not isinstance(y_hat, np.ndarray) or y_hat.shape != y.shape:
		return None
	if not isinstance(pos_label, int) and not isinstance(pos_label, str):
		return None
	try:
		precision = precision_score_(y, y_hat, pos_label)
		recall = recall_score_(y, y_hat, pos_label)
		return 2 * precision * recall / (precision + recall)
	except:
		return None
