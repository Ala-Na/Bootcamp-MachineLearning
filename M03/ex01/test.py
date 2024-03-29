import numpy as np
from log_pred import logistic_predict_

x = np.array([4]).reshape((-1, 1))
theta = np.array([[2], [0.5]])
print(logistic_predict_(x, theta), '\n')
# Output: array([[0.98201379]])

x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
print(logistic_predict_(x2, theta2)), '\n'
# Output: array([[0.98201379],[0.99624161],[0.97340301],[0.99875204],[0.90720705]])

x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
print(logistic_predict_(x3, theta3), '\n')
# Output:array([[0.03916572],[0.00045262],[0.2890505 ]])

print("\n\nCORRECTION:")


x=np.array([0]).reshape(-1, 1)
theta=np.array([[0], [0]])
print(logistic_predict_(x, theta))
print("ans = np.array([[0.5]])")
print()

x=np.array([1]).reshape(-1, 1)
theta=np.array([[1], [1]])
print(logistic_predict_(x, theta))
print("ans = np.array([[0.880797077978]])")
print()

x=np.array([[1, 0], [0, 1]])
theta=np.array([[1], [2], [3]])
print(logistic_predict_(x, theta))
print("ans = np.array([[0.952574126822], [0.982013790038]])")
print()

x=np.array([[1, 1], [1, 1]])
theta=np.array([[1], [2], [3]])
print(logistic_predict_(x, theta))
print("ans = np.array([[0.997527376843], [0.997527376843]])")
