import numpy as np
from plot import *

# BE CAREFUL ! In subject, given example don't have the good value for loss and
# perform :
# np.sum(((y - y_hat) ** 2) / y.shape[0])
# instead of :
# np.sum(((y - y_hat) ** 2) / (2 * y.shape[0]))

x = np.arange(1,6).reshape(-1, 1)
y = np.array([[11.52434424],[10.62589482],[13.14755699],[18.60682298],[14.14329568]])
theta1= np.array([[18],[-1]])
plot_with_loss(x, y, theta1)

theta2 = np.array([[14],[0]])
plot_with_loss(x, y, theta2)

theta3 = np.array([[12],[0.8]])
plot_with_loss(x, y, theta3)
