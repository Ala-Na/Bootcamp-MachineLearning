import numpy as np

def united_shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(proportion, float) or proportion > 1 or proportion < 0:
        return None
    try:
        ind_split = (int)(x.shape[0] * proportion)
        x, y = united_shuffle(x, y)
        return (x[:ind_split, :], x[ind_split:, :], y[:ind_split, :], y[ind_split:, :])
    except:
        return None
    
