import numpy as np
import math

def mse_(y, y_hat):
    '''
    Description:
    Calculate the MSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    '''
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.shape != y.shape:
        return None
    mse = ((y_hat - y) ** 2).mean(axis=None)
    return float(mse)

def rmse_(y, y_hat):
    """
    Description:
    Calculate the RMSE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    rmse: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.shape != y.shape:
        return None
    rmse = math.sqrt(mse_(y, y_hat))
    return float(rmse)

def mae_(y, y_hat):
    """
    Description:
    Calculate the MAE between the predicted output and the real output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    mae: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.shape != y.shape:
        return None
    mae = (np.absolute(y_hat - y)).mean(axis=None)
    return float(mae)

def r2score_(y, y_hat):
    """
    Description:
    Calculate the R2score between the predicted output and the output.
    Args:
    y: has to be a numpy.array, a vector of dimension m * 1.
    y_hat: has to be a numpy.array, a vector of dimension m * 1.
    Returns:
    r2score: has to be a float.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exceptions.
    """
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.shape != y.shape:
        return None
    y_bar = y.mean(axis=None)
    r2score = 1 - (np.sum((y_hat - y) ** 2) / np.sum((y - y_bar) ** 2))
    return float(r2score)
