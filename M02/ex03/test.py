import numpy as np
from gradient import gradient

x = np.array([[ -6, -7, -9],[ 13, -2, 14],[ -7, 14, -1],[ -8, -4, 6],[ -5, -9, 6],[ 1, -5, 11],[ 9, -11, 8]])
y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))


# /!\ Thetas here are not those of the subject, a 0 is added in first position of
# the numpy array to fit thetas conditions of gradient function :
# Theta must be (n + 1) * 1 for an x of m * n dimensions.

# In subject examples, thetas are only 3 * 1 for x of 7 * 3 dimensions...

theta1 = np.array([0, 3,0.5,-6]).reshape((-1, 1))
print(gradient(x, y, theta1), '\n')
# Output: array([[ -33.71428571],[ -37.35714286],[ 183.14285714],[ -393.]])

theta2 = np.array([0, 0,0,0]).reshape((-1, 1))
print(gradient(x, y, theta2), '\n')
# Output: array([[ -0.71428571],[ 0.85714286],[ 23.28571429],[ -26.42857143]])

print("Correction follow up")
x = np.ones(10).reshape(-1,1)
theta = np.array([[1], [1]])
y = np.ones(10).reshape(-1,1)
print(f"{gradient(x, y, theta) = }")
print("array([[1], [1]])")

x = (np.arange(1,25)).reshape(-1,2)
theta = np.array([[3],[2],[1]])
y = np.arange(1,13).reshape(-1,1)
# print(f"{x.shape = }")
# print(f"{y.shape = }")
# print(f"{theta.shape = }")
print(f"{gradient(x, y, theta) = }")
print("""array([[ 33.5       ],
	[521.16666667],
	[554.66666667]])""")

x = (np.arange(1,13)).reshape(-1,3)
theta = np.array([[5],[4],[-2],[1]])
y = np.arange(9,13).reshape(-1,1)
print(f"{gradient(x, y, theta) = }")
print("array([[ 11. ], [ 90.5], [101.5], [112.5]])")
