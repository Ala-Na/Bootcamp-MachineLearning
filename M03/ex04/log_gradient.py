import math
import numpy as np

def log_gradient(x, y, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.size == 0:
        return None
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1 or y.shape[0] != x.shape[0] or y.size == 0:
        return None
    if not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[1] != 1 or theta.shape[0] != x.shape[1] + 1 or theta.size == 0:
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    gradient = np.sum(1 / (1 + np.exp(np.dot(-X, theta))) - y) / x.shape[0]
    for idx in range(1, theta.shape[0]):
        grad_j = 0
        for i in range(0, X.shape[0]):
            grad_j += (1 / (1 + np.exp(np.dot(-X[i], theta))) - y[i]) * X[i, idx]
        grad_j = grad_j / x.shape[0]
        gradient = np.append(gradient, grad_j)
    return gradient.reshape(-1, 1)

# Example 1:
y1 = np.array([[1]])
x1 = np.array([[4]])
theta1 = np.array([[2], [0.5]])
print(log_gradient(x1, y1, theta1))
# Output: array([[-0.01798621], [-0.07194484]])
# Example 2:
y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print(log_gradient(x2, y2, theta2))
# Output: array([[0.3715235 ], [3.25647547]])
# Example 3:
y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(log_gradient(x3, y3, theta3))
# Output: array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])