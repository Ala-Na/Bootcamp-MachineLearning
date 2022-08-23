import numpy as np
from log_loss import *

y1 = np.array([1]).reshape((-1, 1))
x1 = np.array([4]).reshape((-1, 1))
theta1 = np.array([[2], [0.5]])
y_hat1 = logistic_predict_(x1, theta1)
print(log_loss_(y1, y_hat1), '\n')
# Output: 0.01814992791780973

y2 = np.array([[1], [0], [1], [0], [1]])
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
theta2 = np.array([[2], [0.5]])
y_hat2 = logistic_predict_(x2, theta2)
print(log_loss_(y2, y_hat2), '\n')
# Output: 2.4825011602474483

y3 = np.array([[0], [1], [1]])
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
y_hat3 = logistic_predict_(x3, theta3)
print(log_loss_(y3, y_hat3))
# Output: 2.9938533108607053

print("CORRECTION:")
y=np.array([[0], [0]])
y_hat=np.array([[0], [0]])
print(f"{log_loss_(y, y_hat) = }")
print("Ans = 0 / close to 0")
print()

y=np.array([[0], [1]])
y_hat=np.array([[0], [1]])
print(f"{log_loss_(y, y_hat) = }")
print("Ans = 0 / close to 0")
print()

y=np.array([[0], [0], [0]])
y_hat=np.array([[1], [0], [0]])
print(f"{log_loss_(y, y_hat) = }")
print("Ans = 11.51292546")
print()

y=np.array([[0], [0], [0]])
y_hat=np.array([[1], [0], [1]])
print(f"{log_loss_(y, y_hat) = }")
print("Ans = 23.02585093")
print()

y=np.array([[0], [1], [0]])
y_hat=np.array([[1], [0], [1]])
print(f"{log_loss_(y, y_hat) = }")
print("Ans = 34.53877639")
print()
