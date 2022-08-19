import numpy as np
from prediction import predict_

x = np.arange(1,13)
x = np.reshape(x, (4,3))


x = np.arange(1,13).reshape((4,-1))
# Example 1:
theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
print(predict_(x, theta1), '\n')
# Ouput:
#array([[5.], [5.], [5.], [5.]])
# Do you understand why y_hat contains only 5’s here?
# Example 2:
theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
print(predict_(x, theta2), '\n')
# Output:
#array([[ 1.], [ 4.], [ 7.], [10.]])
# Do you understand why y_hat == x[:,0] here?
# Example 3:
theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
print(predict_(x, theta3), '\n')
# Output:
#array([[ 9.64], [24.28], [38.92], [53.56]])
# Example 4:
theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
print(predict_(x, theta4), '\n')
# Output:
#array([[12.5], [32. ], [51.5], [71. ]])

y = [[]]
print(predict_(y, theta1))

theta5 = [[]]
print(predict_(y, theta5))
print(predict_(x, theta5))

theta6 = np.array([-3, 1, 2]).reshape((-1, 1))
print(predict_(x, theta6))

print("CORRECTION:")
print("Test 1")
x = (np.arange(1,13)).reshape(-1,2)
theta = np.ones(3).reshape(-1,1)
print(predict_(x, theta))
print("array([[ 4.], [ 8.], [12.], [16.], [20.], [24.]])")
print()

print("Test 2")
x = (np.arange(1,13)).reshape(-1,3)
theta = np.ones(4).reshape(-1,1)
print(predict_(x, theta))
print("array([[ 7.], [16.], [25.], [34.]])")
print()

print("Test 3")
x = (np.arange(1,13)).reshape(-1,4)
theta = np.ones(5).reshape(-1,1)
print(predict_(x, theta))
print("array([[11.], [27.], [43.]])")
print()
