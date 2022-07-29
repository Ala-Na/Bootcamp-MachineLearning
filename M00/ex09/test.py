import numpy as np
from other_losses import *

x = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
print(mse_(x,y))
print(rmse_(x,y))
print(mae_(x,y))
print(r2score_(x, y))
