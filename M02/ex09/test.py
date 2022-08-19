
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

print("\nCORRECTION:")
x = np.ones(42).reshape((-1, 1))
y = np.ones(42).reshape((-1, 1))
ret = data_spliter(x, y, 0.42)
print(list(map(np.shape, ret)))
print("[(17,1), (25,1), (17,1), (25,1)]")


np.random.seed(42)
tmp= np.arange(0,110).reshape(11,10)
x = tmp[:, :-1]
y = tmp[:,-1].reshape((-1, 1))
ret = data_spliter(x, y, 0.42)
print(ret)
print("""(array([[ 50,  51,  52,  53,  54,  55,  56,  57,  58],
	[  0,   1,   2,   3,   4,   5,   6,   7,   8],
	[ 90,  91,  92,  93,  94,  95,  96,  97,  98],
	[100, 101, 102, 103, 104, 105, 106, 107, 108]]), array([[20, 21, 22, 23, 24, 25, 26, 27, 28],
	[10, 11, 12, 13, 14, 15, 16, 17, 18],
	[80, 81, 82, 83, 84, 85, 86, 87, 88],
	[40, 41, 42, 43, 44, 45, 46, 47, 48],
	[70, 71, 72, 73, 74, 75, 76, 77, 78],
	[30, 31, 32, 33, 34, 35, 36, 37, 38],
	[60, 61, 62, 63, 64, 65, 66, 67, 68]]), array([[ 59],
	[  9],
	[ 99],
	[109]]), array([[29],
	[19],
	[89],
	[49],
	[79],
	[39],
	[69]]))""")
print(list(map(np.shape, ret)))
print("[(4, 9), (7, 9), (4,1), (7,1)]")
