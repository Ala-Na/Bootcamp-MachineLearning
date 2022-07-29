import numpy as np

def simple_predict(x, theta):
    '''Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    '''
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != 1:
        print("x is not a non-empty numpy array of dimension m * 1")
        return None
    elif not isinstance(theta, np.ndarray) or not np.issubdtype(theta.dtype, np.number) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return None
    try:
        res = np.zeros((x.shape[0], 1))
        for ind in range(x.shape[0]):
            value = theta[0][0] + theta[1][0] * x[ind][0]
            res[ind][0] = value
        return res
    except:
        print("Something went wrong")
        return res

# Interlude - Predict, Evaluate, Improve
# A computer program is said to learn from experience E, with respect to some class of
# tasks T, and performance measure P, if its performance at tasks in T, as measured by P,
# improves with experience E. (Tom Mitchell, Machine Learning, 1997)
# In other words to learn you have to improve.
# To improve you have to evaluate your performance.
# To evaluate your performance you need to start performing on the task you want to be
# good at.
# One of the most common tasks in Machine Learning is prediction.
# This will be your algorithm’s task.
# This will be your task.

# Predict
# A very simple model
# We have some data. We want to model it.
# • First we need to make an assumption, or hypothesis, about the structure of the data
# and the relationship between the variables.
# • Then we can apply that hypothesis to our data to make predictions.
# hypothesis(data) = predictions
# Hypothesis
# Let’s start with a very simple and intuitive hypothesis on how the price of a spaceship
# can be predicted based on the power of its engines.
# We will consider that the more powerful the engines are, the more expensive the spaceship
# is.
# Furthermore, we will assume that the price increase is proportional to the power in-
# crease. In other words, we will look for a linear relationship between the two vari-
# ables.
# This means that we will formulate the price prediction with a linear equation, that you
# might be already familiar with:
# ˆy = ax + b
# We add the ˆ symbol over the y to specify that ˆy (pronounced y-hat) is a prediction (or
# estimation) of the real value of y. The prediction is calculated with the parameters a
# and b and the input value x.
# For example, if a = 5 and b = 33, then ˆy = 5x + 33.
# But in Machine Learning, we don’t like using the letters a and b. Instead we will use the
# following notation:
# y = θ0 + θ1x
# So if θ0 = 33 and θ1 = 5, then ˆy = 33 + 5x.
# To recap, this linear equation is our hypothesis. Then, all we will need to do is find
# the right values for our parameters θ0 and θ1 and we will get a fully-functional prediction
# model.
# Predictions
# Now, how can we generate a set of predictions on an entire dataset? Let’s consider a
# dataset containing m data points (or space ships), called examples.
# What we do is stack the x and ˆy values of all examples in vectors of length m. The
# relation between the elements in our vectors can then be represented with the following
# formula:
# ˆy(i) = θ0 + θ1x(i) for i = 1, ..., m
# Where:
# • ˆy(i) is the ith component of vector y
# • x(i) is the ith component of vector x

# More information
# Why the θ notation?
# You might have two questions at the moment:
# • WTF is that weird symbol? This strange symbol, θ, is called "theta".
# • Why use this notation instead of a and b, like we’re used to? Despite its
# seeming more complicated at first, the theta notation is actually meant to simplify
# your equations later on. Why? a and b are good for a model with two parameters,
# but you will soon need to build more complex models that take into account more
# variables than just x. You could add more letters like this: ˆy = ax1 + bx2 + cx3 +
# ... + yx25 + z But how do you go beyond 26 parameters? And how easily can
# you tell what parameter is associated with, let’s say, x19? That’s why it becomes
# more handy to describe all your parameters using the theta notation and indices.
# With θ, you just have to increment the number to name the parameter: ˆy =
# θ0 + θ1x1 + θ2x2 + ... + θ2468x2468 ... Easy right?
# Another common notation
# ˆy = hθ(x)
# Because ˆy is calculated with our linear hypothesis using θ and x, it is sometimes written
# as hθ(x). The h stands for hypothesis, and can be read as "the result of our hypothesis h
# given x and theta".
# Then if x = 7, we can calculate: ˆy = hθ(x) = 33 + 5 × 7 = 68 We can now say that
# according to our linear model, the predicted value of y given (x = 7) is 68.
