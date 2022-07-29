import numpy as np
import matplotlib.pyplot as plt

def predict_(x, theta):
    '''Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    '''
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        print("x is not a non-empty numpy array of dimension m * 1")
        return None
    elif not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return None
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot(X, theta)
    except:
        print("something went wrong")
        return None

def loss_(y, y_hat):
    '''Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    '''
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.shape != y.shape:
        return None
    try:
        return np.sum(((y - y_hat) ** 2) / (2 * y.shape[0]))
    except:
        print("Something went wrong")
        return None


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        print("x is not a numpy array of dimension m * 1")
        return
    elif not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        print("y is not a numpy array of dimension m * 1")
        return
    elif not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return
    y_hat = predict_(x, theta)
    cost = loss_(y, y_hat)
    plt.title("Cost = {:.06f}".format(cost))
    plt.plot(x, y, 'o')
    plt.plot(x, y_hat)
    for i in range(x.shape[0]):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], 'r--')
    plt.show()
