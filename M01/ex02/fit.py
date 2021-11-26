import numpy as np

def predict(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    elif not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.dot(X, theta)

def gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    X = np.insert(x, 0, 1.0, axis = 1)
    X_T = X.transpose()
    y_hat = np.dot(X, theta)
    return np.dot(X_T, y_hat - y) / x.shape[0]

def fit_(x, y, theta, alpha, max_iter):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    if not isinstance(alpha, float) or not isinstance(max_iter, int):
        return None
    new_theta = theta
    for i in range(0, max_iter):
        curr_gradient = gradient(x, y, new_theta)
        theta_zero = (float)(new_theta[0][0] - alpha * curr_gradient[0][0])
        theta_one = (float)(new_theta[1][0] - alpha * curr_gradient[1][0])
        new_theta = np.asarray([theta_zero, theta_one]).reshape((2, 1))
    return new_theta
        
    

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta = np.array([[1],[ 1]])

theta1 = fit_(x, y, theta, alpha=5e-6, max_iter=15000)
print(theta1)
#array([[1.40709365], [1.1150909 ]])

print("\n")
print(predict(x, theta1))
#array([[15.3408728 ],[25.38243697],[36.59126492],[55.95130097],[65.53471499]])