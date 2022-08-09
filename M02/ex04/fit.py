import numpy as np

def predict_(x, theta):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0:
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1:
        return None
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        return np.dot(X, theta)
    except:
        return None

def gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.shape != (x.shape[1] + 1, 1):
        return None
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        return np.matmul((np.transpose(X) / x.shape[0]), (np.matmul(X, theta) - y))
    except:
        return None


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of dimension m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of dimension m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1):
        return None
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    for i in range(0, max_iter):
        curr_gradient = gradient(x, y, theta)
        new_theta = []
        for val in range(0, theta.shape[0]):
            new_theta.append((float)(theta[val][0] - alpha * curr_gradient[val][0]))
        theta = np.asarray(new_theta).reshape(theta.shape)
    return theta

# • You can create more training data by generating an x array
# with random values and computing the corresponding y vector as
# a linear expression of x. You can then fit a model on this
# artificial data and find out if it comes out with the same θ
# coefficients that first you used.
# • It is possible that θ0 and θ1 become "nan". In that case, it
# means you probably used a learning rate that is too large.
