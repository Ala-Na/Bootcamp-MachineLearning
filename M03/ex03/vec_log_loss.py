import numpy as np

def logistic_predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != x.shape[1] + 1 or theta.shape[1] != 1:
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return 1 / (1 + np.exp(-X @ theta))

#Same thoughts as ex02
def vec_log_loss_(y, y_hat, eps=1e-15):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    if not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1 or y_hat.shape[0] != y.shape[0]:
        return None
    if not isinstance(eps, float):
        return None
    one_vec = np.ones((1, y.shape[1]))
    return (float)(1 / y.shape[0])*(((-y).T @ np.log(np.clip(y_hat, eps, 1 - eps))) - ((one_vec - y).T @ np.log(np.clip(one_vec - y_hat, eps, 1 - eps)))).item()
    #return - (np.sum(y * np.log(eps + y_hat) + (one_vec - y) * np.log(eps + one_vec - y_hat)) / y.shape[0])
    
# Example 1:
y1 = np.array([[1]])
x1 = np.array([[4]])
theta1 = np.array([[2], [0.5]])
y_hat1 = logistic_predict_(x1, theta1)
print(vec_log_loss_(y1, y_hat1, eps=0.0))
# Output: 0.01814992791780973
# Example 2:
y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
y_hat2 = logistic_predict_(x2, theta2)
print(vec_log_loss_(y2, y_hat2, eps=0.0))
# Output: 2.4825011602474483
# Example 3:
y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
y_hat3 = logistic_predict_(x3, theta3)
print(vec_log_loss_(y3, y_hat3, eps=0.0))
# Output: 2.9938533108607053