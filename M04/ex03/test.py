import numpy as np
from logistic_loss_reg import reg_log_loss_

y = np.array([1, 1, 0, 0, 1, 1, 0]).reshape((-1, 1))
y_hat = np.array([.9, .79, .12, .04, .89, .93, .01]).reshape((-1, 1))
theta = np.array([1, 2.5, 1.5, -0.9]).reshape((-1, 1))

# Example :
print(reg_log_loss_(y, y_hat, theta, .5))
# Output:
#0.43377043716475955

# Example :
print(reg_log_loss_(y, y_hat, theta, .05))
# Output:
#0.13452043716475953

# Example :
print(reg_log_loss_(y, y_hat, theta, .9))
# Output:
#0.6997704371647596

print("CORRECTION:")
y = np.array([0, 1, 0, 1]).reshape(-1, 1)
y_hat = np.array([0.4, 0.79, 0.82, 0.04]).reshape(-1, 1)
theta = np.array([5, 1.2, -3.1, 1.2]).reshape(-1, 1)

print(f"{reg_log_loss_(y, y_hat, theta, .5) = }")
print("Ans = 2.2006805525617885")
print()

print(f"{reg_log_loss_(y, y_hat, theta, .75) = }")
print("Ans = 2.5909930525617884")
print()

print(f"{reg_log_loss_(y, y_hat, theta, 1.0) = }")
print("Ans = 2.981305552561788")
print()

print(f"{reg_log_loss_(y, y_hat, theta, 0.0) = }")
print("Ans = 1.4200555525617884")
print()
