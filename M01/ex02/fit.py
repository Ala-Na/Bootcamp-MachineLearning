import numpy as np

def predict(x, theta):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        return None
    elif not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.shape != (2, 1):
        return None
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot(X, theta)
    except:
        return None

def gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.shape != (2, 1):
        return None
    try:
        X = np.insert(x, 0, 1.0, axis = 1)
        X_T = X.transpose()
        y_hat = np.dot(X, theta)
        return np.dot(X_T, y_hat - y) / x.shape[0]
    except:
        return None

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
    new_theta: numpy.ndarray, a vector of dimension 2 * 1.
    None if there is a matching dimension problem.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.shape != (2, 1):
        return None
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    try:
        new_theta = theta
        for i in range(0, max_iter):
            curr_gradient = gradient(x, y, new_theta)
            theta_zero = (float)(new_theta[0][0] - alpha * curr_gradient[0][0])
            theta_one = (float)(new_theta[1][0] - alpha * curr_gradient[1][0])
            new_theta = np.asarray([theta_zero, theta_one]).reshape((2, 1))
        return new_theta
    except:
        return None

# You can create more training data by generating an x array
# with random values and computing the corresponding y vector as
# a linear expression of x. You can then fit a model on this
# artificial data and find out if it comes out with the same θ
# coefficients that first you used.

# It is possible that θ0 and θ1 become "nan". In that case, it
# means you probably used a learning rate that is too large.
