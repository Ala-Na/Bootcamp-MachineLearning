
import numpy as np
from data_spliter import data_spliter

x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

# Example 0:
print(data_spliter(x1, y, 0.8), '\n')
# example output: (array([[ 10],[ 42],[ 1],[ 300]]), array([[59]]), array([[1],[ 1],[ 0],[ 0]]), array([[0]]))

# Example 1:
print(data_spliter(x1, y, 0.5), '\n')
# example output: (array([[42],[ 10]]), array([[ 59],[ 300],[ 1]]), array([[1],[ 1]]), array([[0],[ 0],[ 0]]))

x2 = np.array([[ 1, 42],
[300, 10],
[ 59, 1],
[300, 59],
[ 10, 42]])
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))

# Example 2:
print(data_spliter(x2, y, 0.8), '\n')
# example output:(array([[ 10, 42],[ 59, 1],[ 1, 42],[300, 10]]), array([[300, 59]]), array([[0],[ 0],[ 0],[ 1]]),array([[1]]))

# Example 3:
print(data_spliter(x2, y, 0.5))
# example output:(array([[300, 10],[ 1, 42]]),array([[ 10, 42],[300, 59],[ 59, 1]]),array([[1],[ 0]]),array([[0],[ 1],[ 0]]))
