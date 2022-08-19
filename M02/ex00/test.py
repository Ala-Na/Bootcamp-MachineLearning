from prediction import simple_predict
import numpy as np

x = np.arange(1,13).reshape((4,-1))

theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1))
print(simple_predict(x, theta1), '\n')
# Ouput: array([[5.],[ 5.],[ 5.],[ 5.]])

theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
print(simple_predict(x, theta2), '\n')
# Output: array([[ 1.],[ 4.],[ 7.],[ 10.]])

theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
print(simple_predict(x, theta3), '\n')
# Output: array([[ 9.64],[ 24.28],[ 38.92],[ 53.56]])

theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
print(simple_predict(x, theta4))
# Output: array([[12.5],[ 32. ],[ 51.5],[ 71. ]])

print("\nCORRECTION:")
print("Test 1")
x = (np.arange(1,13)).reshape(-1,2)
theta = np.ones(3).reshape(-1,1)
print(simple_predict(x, theta))
print("array([[ 4.], [ 8.], [12.], [16.], [20.], [24.]])")
print()

print("Test 2")
x = (np.arange(1,13)).reshape(-1,3)
theta = np.ones(4).reshape(-1,1)
print(simple_predict(x, theta))
print("array([[ 7.], [16.], [25.], [34.]])")
print()

print("Test 3")
x = (np.arange(1,13)).reshape(-1,4)
theta = np.ones(5).reshape(-1,1)
print(simple_predict(x, theta))
print("array([[11.], [27.], [43.]])")
print()
