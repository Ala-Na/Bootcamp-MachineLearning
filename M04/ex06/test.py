import numpy as np
from ridge import *

# No example available in subject

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

ridge = MyRidge(theta)

print(ridge.get_params_())

print('\n')
#print(ridge.set_params_(thetas=[]))

print(ridge.set_params_(alpha=0.0001, max_iter=15000, lambda_=1.), '\n')

print('Check gradient from previous exercice:', ridge.gradient_(x, y).reshape(1, -1), '\n')
# Output:
#array([[ -60.99 ],
#[-195.64714286],
#[ 863.46571429],
#[-644.52142857]])

y_hat = ridge.predict_(x)
print('Predict before fit:', y_hat.reshape(1, -1))
print('Loss elem before fit:', ridge.loss_elem_(y, y_hat).reshape(1, -1))
print('Loss before fit:', ridge.loss_(y, y_hat), '\n')

ridge.fit_(x, y)

y_hat = ridge.predict_(x)
print('Predict after fit:', y_hat.reshape(1, -1))
print('Loss elem after fit:', ridge.loss_elem_(y, y_hat).reshape(1, -1))
print('Loss after fit:', ridge.loss_(y, y_hat), '\n')
