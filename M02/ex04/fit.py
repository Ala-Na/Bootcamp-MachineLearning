import numpy as np

def predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(theta, np.ndarray) or theta.shape[0] != x.shape[1] + 1:
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.dot(X, theta)

def gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(y, np.ndarray) or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(theta, np.ndarray) or theta.shape != (x.shape[1] + 1, 1):
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.matmul((np.transpose(X) / x.shape[0]), (np.matmul(X, theta) - y))

def fit_(x, y, theta, alpha, max_iter):
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

x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])

theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
print(theta2, '\n')
# Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])

print(predict_(x, theta2))
# Output: array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]]