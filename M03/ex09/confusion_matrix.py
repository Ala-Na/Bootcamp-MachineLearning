import numpy as np
import pandas as pd

def confusion_matrix_(y_true, y_hat, labels=None, df_option=False):
	"""
	Compute confusion matrix to evaluate the accuracy of a classification.
	Args:
	y:a numpy.array for the correct labels
	y_hat:a numpy.array for the predicted labels
	labels: optional, a list of labels to index the matrix.
	This may be used to reorder or select a subset of labels. (default=None)
	df_option: optional, if set to True the function will return a pandas DataFrame
	instead of a numpy array. (default=False)
	Return:
	The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
	None if any error.
	Raises:
	This function should not raise any Exception.
	"""
	if not isinstance(y_true, np.ndarray) or y_true.size == 0:
		return None
	if not isinstance(y_hat, np.ndarray) or y_hat.shape != y_true.shape:
		return None
	if labels is not None and not isinstance(labels, list):
		return None
	if not isinstance(df_option, bool):
		return None
	try:
		if labels is None:
			y_total = np.stack((y_true, y_hat))
			labels = np.unique(y_total)
		confusion_matrix = np.zeros((len(labels), len(labels)))
		for i in range(len(labels)):
			for j in range(len(labels)):
				confusion_matrix[i, j] = np.sum((y_true == labels[i]) & (y_hat == labels[j]))
		if df_option:
			return pd.DataFrame(confusion_matrix, index=labels, columns=labels)
		else:
			return confusion_matrix
	except:
		return None
