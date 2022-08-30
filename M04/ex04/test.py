import numpy as np
from reg_linear_grad import *

x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[ -8, -4, 6],
[ -5, -9, 6],
[ 1, -5, 11],
[ 9, -11, 8]])
y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
theta = np.array([[7.01], [3], [10.5], [-6]])

# MODIFICATION WITH SUBJECT : Because lambda_ has to be a float and not an int !

# Example 1.1:
print(reg_linear_grad(y, x, theta, 1.0))
# Output:
#array([[ -60.99 ],
#[-195.64714286],
#[ 863.46571429],
#[-644.52142857]])

# Example 1.2:
print(vec_reg_linear_grad(y, x, theta, 1.0))
# Output:
#array([[ -60.99 ],
#[-195.64714286],
#[ 863.46571429],
#[-644.52142857]])

# Example 2.1:
print(reg_linear_grad(y, x, theta, 0.5))
# Output:
#array([[ -60.99 ],
#[-195.86142857],
#[ 862.71571429],
#[-644.09285714]])

# Example 2.2:
print(vec_reg_linear_grad(y, x, theta, 0.5))
# Output:
#array([[ -60.99 ],
#[-195.86142857],
#[ 862.71571429],
#[-644.09285714]])

# Example 3.1:
print(reg_linear_grad(y, x, theta, 0.0))
# Output:
#array([[ -60.99 ],
#[-196.07571429],
#[ 861.96571429],
#[-643.66428571]])

# Example 3.2:
print(vec_reg_linear_grad(y, x, theta, 0.0))
# Output:
#array([[ -60.99 ],
#[-196.07571429],
#[ 861.96571429],
#[-643.66428571]])

print("\nCORRECTION:")
x = np.arange(7,49).reshape(7,6)
y = np.array([[1], [1], [2], [3], [5], [8], [13]]).reshape(-1, 1)
theta = np.array([[16], [8], [4], [2], [0], [0.5], [0.25]]).reshape(-1, 1)

print(f"{vec_reg_linear_grad(y, x, theta, 0.5) = }")
print("""array([[  391.28571429],
		[11861.28571429],
		[12252.28571429],
		[12643.42857143],
		[13034.57142857],
		[13425.89285714],
		[13817.16071429]])""")
print()

print(f"{vec_reg_linear_grad(y, x, theta, 1.5) = }")
print("""array([[  391.28571429],
		[11862.42857143],
		[12252.85714286],
		[12643.71428571],
		[13034.57142857],
		[13425.96428571],
		[13817.19642857]])""")
print()

print(f"{vec_reg_linear_grad(y, x, theta, 0.05) = }")
print("""array([[  391.28571429],
		[11860.77142857],
		[12252.02857143],
		[12643.3       ],
		[13034.57142857],
		[13425.86071429],
		[13817.14464286]])""")
print()
