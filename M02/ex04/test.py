import numpy as np
from fit import *

x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
theta = np.array([[42.], [1.], [1.], [1.]])

theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42000)
print(theta2)
print("Output: array([[41.99..],[0.97..], [0.77..], [-1.20..]])\n")

print(predict_(x, theta2))
print("Output: array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])\n")

print("CORRECTION: ")
x = np.arange(1,13).reshape(-1,3)
y = np.arange(9,13).reshape(-1,1)
theta = np.array([[5], [4], [-2], [1]])
alpha = 1e-2
max_iter = 10000
print(f"{fit_(x, y, theta, alpha = alpha, max_iter=max_iter)}")
print(f"Answer = array([[ 7.111..],[ 1.0],[-2.888..],[ 2.222..]])")

x = np.arange(1,31).reshape(-1,6)
theta = np.array([[4],[3],[-1],[-5],[-5],[3],[-2]])
y = np.array([[128],[256],[384],[512],[640]])
alpha = 1e-4
max_iter = 42000
print(f"\n{fit_(x, y, theta, alpha=alpha, max_iter=max_iter)}")
print(f"""Answer = array([[ 7.01801797]
	[ 0.17717732]
	[-0.80480472]
	[-1.78678675]
	[ 1.23123121]
	[12.24924918]
	[10.26726714]])""")
