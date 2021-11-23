import numpy as np

def predict_(x, theta):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        print("x is not a numpy array of dimension m * 1")
        return None
    elif not isinstance(theta, np.ndarray) or theta.ndim != 2 or theta.shape[0] != 2 or theta.shape[1] != 1:
        print("theta is not a numpy array of dimensions 2 * 1")
        return None
    X = np.insert(x, 0, 1.0, axis=1)
    return np.dot(X, theta)

def loss_elem_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1:
        return None        
    J_elem = []
    for yi, yi_hat in zip(y, y_hat):
        J_elem.append([(yi_hat[0] - yi[0]) ** 2])
    return np.array(J_elem)

def loss_(y, y_hat):
    if not isinstance(y, np.ndarray) or y.ndim != 2 or y.shape[1] != 1:
        return None
    elif not isinstance(y_hat, np.ndarray) or y_hat.ndim != 2 or y_hat.shape[1] != 1:
        return None
    J_elem = loss_elem_(y, y_hat)
    J_value = float(1/(2*y.shape[0]) * np.sum(J_elem))
    return J_value

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
print(loss_elem_(y1, y_hat1))
print(loss_(y1, y_hat1))

x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])
print(loss_elem_(y2, y_hat2))
print(loss_(y2, y_hat2))

x3 = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(loss_(y3, y_hat3))
print(loss_(y3, y3))