import numpy as np
from linear_loss_reg import reg_loss_

y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
y_hat = np.array([3, 13, -11.5, 5, 11, 5, -20]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

# Example :
print(reg_loss_(y, y_hat, theta, .5))
# Output:
#0.8503571428571429

# Example :
print(reg_loss_(y, y_hat, theta, .05))
# Output:
#0.5511071428571429

# Example :
print(reg_loss_(y, y_hat, theta, .9))
# Output:
#1.116357142857143

print("\nCORRECTION:")
y=np.arange(10,100,10).reshape(-1, 1)
y_hat=np.arange(9.5,95,9.5).reshape(-1, 1)
theta=np.array([-10,3,8]).reshape(-1, 1)
lambda_=0.5
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 5.986111111111111")
print()

lambda_ = 5.0
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 24.23611111111111")
print()

y = np.arange(-15,15,0.1).reshape(-1, 1)
y_hat = np.arange(-30,30,0.2).reshape(-1, 1)
theta=np.array([42,24,12]).reshape(-1, 1)
lambda_=0.5
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 38.10083333333307")
print()

lambda_=8.0
print(f"{reg_loss_(y, y_hat, theta, lambda_) = }")
print("Ans = 47.10083333333307")
print()
