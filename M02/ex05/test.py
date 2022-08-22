import numpy as np

from mylinearregression import MyLinearRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
Y = np.array([[23.], [48.], [218.]])

mylr = MyLR([[1.], [1.], [1.], [1.], [1]])

# Example 0:
Y_hat = mylr.predict_(X)
print(Y_hat)
print("Output:array([[8.], [48.], [323.]])\n")

# Example 1:
print(mylr.loss_elem_(Y, Y_hat))
print("Output:array([[225.], [0.], [11025.]])\n")

# Example 2:
print(mylr.loss_(Y, Y_hat))
print("Output:1875.0\n")

print('Can take time...\n')

# Example 3:
mylr.alpha = 1.6e-4
mylr.max_iter = 200000
mylr.fit_(X, Y)
print(mylr.thetas)
print("Output:array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])\n")

# Example 4:
Y_hat = mylr.predict_(X)
print( Y_hat)
print("Output:array([[23.417..], [47.489..], [218.065...]])\n")

# Example 5:
print(mylr.loss_elem_(Y, Y_hat))
print("Output:array([[0.174..], [0.260..], [0.004..]])\n")

# Example 6:
print(mylr.loss_(Y, Y_hat))
print("Output: 0.0732\n")

print("\nCorrection:")

x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
lr1 = MyLR([2, 0.7])


# Example 0.0:
print("# Example 0.0:")
print(lr1.predict_(x))
# Output:
print("""
array([[10.74695094],
[17.05055804],
[24.08691674],
[36.24020866],
[42.25621131]])
""")


# Example 0.1:
print("# Example 0.1:")
print(lr1.loss_elem_(lr1.predict_(x),y))
# Output:
print("""
array([[710.45867381],
	[364.68645485],
	[469.96221651],
	[108.97553412],
	[299.37111101]])
""")

# Example 0.2:
print("# Example 0.2:")
print(lr1.loss_(lr1.predict_(x),y))
# Output:
print(195.34539903032385)


# Example 1.0:
print("# Example 1.0:")
lr2 = MyLR(thetas=[1, 1], alpha=5e-8, max_iter=1500000)
lr2.fit_(x, y)
print(lr2.thetas)
# Output:
print("""
array([[1.40709365],
[1.1150909 ]])
""")


# import sys
# sys.exit()

# Example 1.1:
print("# Example 1.1:")
print(lr2.predict_(x))
# Output:
print("""
array([[15.3408728 ],
[25.38243697],
[36.59126492],
[55.95130097],
[65.53471499]])
""")


# Example 1.2:
print("# Example 1.2:")
print(lr2.loss_elem_(y, lr2.predict_(x)))
# Output:
print("""
array([[486.66604863],
	[115.88278416],
	[ 84.16711596],
	[ 85.96919719],
	[ 35.71448348]])
""")


# Example 1.3:
print("# Example 1.3:")
print(lr2.loss_(y, lr2.predict_(x)))
# Output:
print("80.83996294128525")
