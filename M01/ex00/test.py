import numpy as np
from gradient import simple_gradient

x = np.array([[12.4956442],[ 21.5007972],[ 31.5527382],[ 48.9145838],[ 57.5088733]])
y = np.array([[37.4013816],[ 36.1473236],[ 45.7655287],[ 46.6793434],[ 59.5585554]])
theta1 = np.array([[2],[ 0.7]])
print(simple_gradient(x, y, theta1))
#array([[-19.0342574],[-586.66875564])

print("\n")
theta2 = np.array([[1],[ -0.4]])
print(simple_gradient(x, y, theta2))
#array([[-57.86823748],[-2230.12297889]])

print("\n")
# Error check theta not good shape
theta2 = np.array([[1]])
print(simple_gradient(x, y, theta2))
#None

print("\n")
# Error check x is empty
x=np.array([[]])
print(simple_gradient(x, y, theta2))
#None

print("\n")
# Error check y not good shape
x = np.array([[12.4956442],[ 21.5007972],[ 31.5527382],[ 48.9145838],[ 57.5088733]])
y = np.array([[37.4013816],[ 36.1473236],[ 45.7655287],[ 46.6793434]])
theta2 = np.array([[1],[ -0.4]])
print(simple_gradient(x, y, theta2))
#None
