
import numpy as np
from prediction import simple_predict

print(simple_predict.__doc__)

x = np.arange(1,6).reshape(-1, 1)
print(x)
theta1 = np.array([[5],[0]])
res = simple_predict(x, theta1)
print(res)
theta2 = np.array([[0],[1]])
res = simple_predict(x, theta2)
print(res)
theta3 = np.array([[5],[3]])
res = simple_predict(x, theta3)
print(res)
theta4 = np.array([[-3],[1]])
res = simple_predict(x, theta4)
print(res)
res = simple_predict(np.array([[]]), theta1)
print(res)
