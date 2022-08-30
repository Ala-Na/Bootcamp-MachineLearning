import numpy as np
from l2_reg import *

x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
# Example 1:
print(iterative_l2(x))
# Output:
#911.0
# Example 2:
print(l2(x))
# Output:
#911.0

y = np.array([3,0.5,-6]).reshape((-1, 1))

# Example 3:
print(iterative_l2(y))
# Output:
#36.25
# Example 4:
print(l2(y))
# Output:
#36.25

print("\nCORRECTION:")

theta = np.ones(10).reshape(-1, 1)
print(f"{l2(theta) = }")
print("Ans = 9.0")
print()

theta = np.arange(1, 10).reshape(-1, 1)
print(f"{l2(theta) = }")
print("Ans = 284.0")
print()

theta = np.array([50, 45, 40, 35, 30, 25, 20, 15, 10,  5,  0]).reshape(-1, 1)
print(f"{l2(theta) = }")
print("Ans = 7125.0")
print()
