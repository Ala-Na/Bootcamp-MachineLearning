import numpy as np
from vec_gradient import gradient

x = np.array([[12.4956442],[ 21.5007972],[ 31.5527382],[ 48.9145838],[ 57.5088733]])
y = np.array([[37.4013816],[ 36.1473236],[ 45.7655287],[ 46.6793434],[ 59.5585554]])

theta1 = np.array([[2],[ 0.7]])
print(gradient(x, y, theta1))
print("\n")
#array([[-19.0342574],[-586.66875564])


theta2 = np.array([[1],[ -0.4]])
print(gradient(x, y, theta2))
#array([[-57.86823748],[-2230.12297889]])

print("\n")
# Error check theta not good shape
theta2 = np.array([[1]])
print(gradient(x, y, theta2))
#None

print("\n")
# Error check x is empty
x=np.array([[]])
print(gradient(x, y, theta2))
#None

print("\n")
# Error check y not good shape
x = np.array([[12.4956442],[ 21.5007972],[ 31.5527382],[ 48.9145838],[ 57.5088733]])
y = np.array([[37.4013816],[ 36.1473236],[ 45.7655287],[ 46.6793434]])
theta2 = np.array([[1],[ -0.4]])
print(gradient(x, y, theta2))
#None

def unit_test(n, theta, answer, f):
		x = np.array(range(1,n+1)).reshape((-1, 1))
		y = f(x)
		print(f"Student:\n{gradient(x, y, theta)}")
		print(f"Truth  :\n{answer}")
		print()


theta = np.array([[1.],[1.]])
answer = np.array([[-11.625], [-795.375]])
unit_test(100, theta, answer, lambda x:1.25 * x)

answer = np.array([[-124.125], [-82957.875]])
unit_test(1000, theta, answer, lambda x:1.25 * x)

answer = np.array([[-1.24912500e+03], [-8.32958288e+06]])
unit_test(10000, theta, answer, lambda x:1.25 * x)

theta = np.array([[4], [-1]])
answer = np.array([[-13.625], [-896.375]])
unit_test(100, theta, answer, lambda x:-0.75 * x + 5)

answer = np.array([[-126.125], [-83958.875]])
unit_test(1000, theta, answer, lambda x:-0.75 * x + 5)

answer = np.array([[-1.25112500e+03], [-8.33958388e+06]])
unit_test(10000, theta, answer, lambda x:-0.75 * x + 5)
