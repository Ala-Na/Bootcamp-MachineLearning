import numpy as np
from loss import loss_

X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
Z = np.array([2, 14, -13, 5, 12, 4]).reshape((-1, 1))
A = np.array([2, 14, -13, 5, 12, 4]).reshape((2, -1)) 

print(loss_(X, Y), '\n')
#Output : 2.1428571428571436

print(loss_(X, X), '\n')
#Output : 0

print(loss_(Z,X), '\n')
print(loss_(A,A), '\n')

