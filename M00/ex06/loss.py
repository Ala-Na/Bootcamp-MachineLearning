import numpy as np

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


def loss_elem_(y, y_hat):
    '''
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    '''
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape[1] != 1 or y_hat.shape[0] == 0:
        return None
    try:
        J_elem = []
        for yi, yi_hat in zip(y, y_hat):
            J_elem.append([(yi_hat[0] - yi[0]) ** 2])
        return np.array(J_elem)
    except:
        print("Something went wrong")
        return None

def loss_(y, y_hat):
    '''
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    '''
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] == 0:
        return None
    elif not isinstance(y_hat, np.ndarray) or not np.issubdtype(y_hat.dtype, np.number) or y_hat.ndim != 2 or y_hat.shape[1] != 1 or y_hat.shape[0] == 0:
        return None
    try:
        J_elem = loss_elem_(y, y_hat)
        J_value = float(1/(2*y.shape[0]) * np.sum(J_elem))
        return J_value
    except:
        print("Something went wrong")
        return None

# Introducing the loss function
# How good is our model? It is hard to say just by looking at the plot. We can clearly
# observe that certain regression lines seem to fit the data better than others, but it would
# be convenient to find a way to measure it.

# To evaluate our model, we are going to use a metric called loss function (sometimes
# called cost function). The loss function tells us how bad our model is, how much it
# costs us to use it, how much information we lose when we use it. If the model is good,
# we won’t lose that much, if it’s terrible, we have a high loss!
# The metric you choose will deeply impact the evaluation (and therefore also the training)
# of your model.
# A frequent way to evaluate the performance of a regression model is to measure the
# distance between each predicted value (ˆy(i)) and the real value it tries to predict (y(i)).
# The distances are then squared, and averaged to get one single metric, denoted J

# This loss function is very close to the one called "Mean Squared
# Error", which is frequently mentioned in Machine Learning resources.
# The difference is in the denominator as you can see in the formula of
# the M SE = 1
# m
# ∑m
# i=1(ˆy(i) − y(i))2.
# Except the division by 2m instead of m, these functions are
# rigourously identical: J(θ) = M SE
# 2 .
# MSE is called like that because it represents the mean of the errors
# (i.e.: the differences between the predicted values and the true
# values), squared.
# You might wonder why we choose to divide by two instead of simply
# using the MSE? (It’s a good question, by the way.)
# • First, it does not change the overall model evaluation: if all
# performance measures are divided by two, we can still compare
# different models and their performance ranking will remain the
# same.
# • Second, it will be convenient when we will calculate the
# gradient tomorow.
