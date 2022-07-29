import numpy as np
import matplotlib.pyplot as plt

def plot(x, y, theta):
    '''Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    '''
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        print("x is not a numpy array of dimension m * 1")
        return
    elif not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] != x.shape[0]:
        print("y is not a numpy array of dimension m * 1")
        return
    elif not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return
    try:
        X = np.insert(x, 0, 1.0, axis=1)
        hyp = np.dot(X, theta)
        plt.plot(x, y, 'o')
        plt.plot(x, hyp)
        plt.show()
    except:
        print("Something went wrong")
        return None

# For you information, the task we are performing here is called
# regression. It means that we are trying to predict a continuous
# numerical attribute for all examples (like a price, for instance).
# Later in the bootcamp, you will see that we can predict other things
# such as categories.
