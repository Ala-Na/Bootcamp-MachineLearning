import numpy as np

# NO FOR LOOP

def simple_gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (2, 1):
        return None
    X = np.insert(x, 0, 1.0, axis = 1)
    y_hat = np.dot(X, theta)
    j0 = np.sum(y_hat - y) / x.shape[0]
    j1 = np.sum((y_hat - y) * x) / x.shape[0]
    return np.asarray([j0, j1]).reshape((2, 1))

    #Should return gradient = vector of shape 2 * 1

x = np.array([[12.4956442],[ 21.5007972],[ 31.5527382],[ 48.9145838],[ 57.5088733]])
y = np.array([[37.4013816],[ 36.1473236],[ 45.7655287],[ 46.6793434],[ 59.5585554]])
theta1 = np.array([[2],[ 0.7]])
print(simple_gradient(x, y, theta1))
#array([[-19.0342574],[-586.66875564])

print("\n")
theta2 = np.array([[1],[ -0.4]])
print(simple_gradient(x, y, theta2))
#array([[-57.86823748],[-2230.12297889]])