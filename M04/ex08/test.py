from my_logistic_regression import MyLogisticRegression as mylogr
import numpy as np

theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

# Example 1:
model1 = mylogr(theta, lambda_=5.0)
print(model1.penalty)
# Output
#’l2’

print(model1.lambda_)
# Output
#5.0

# Example 2:
model2 = mylogr(theta, penalty=None)
print(model2.penalty)
# Output
#None

print(model2.lambda_)
# Output
#0.0

# Example 3:
model3 = mylogr(theta, penalty=None, lambda_=2.0)

print(model3.penalty)
# Output
#None

print(model3.lambda_)
# Output
#0.0


x = np.array([[0, 2, 3, 4],
[2, 4, 5, 5],
[1, 3, 2, 7]])
y = np.array([[0], [1], [1]])
theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

model4 = mylogr(theta, penalty='l2', lambda_=1.0)

# Example 4:
print(model4.gradient_(x, y))
# Output:
#array([[-0.55711039],
#[-1.40334809],
#[-1.91756886],
#[-2.56737958],
#[-3.03924017]])
