import numpy as np

def sigmoid_(x):
    if not isinstance(x, np.ndarray):
        return None 
    if x.ndim != 0 and x.ndim != 2:
        return None
    if x.ndim == 2 and x.shape[1] != 1:
        return None
    return 1/(1 + np.exp(-x))
    

x = np.array(-4)
print(sigmoid_(x))
# Output: array([[0.01798620996209156]])

x = np.array(2)
print(sigmoid_(x))
# Output: array([[0.8807970779778823]])

x = np.array([[-4], [2], [0]])
print(sigmoid_(x))
# Output: array([[0.01798620996209156], [0.8807970779778823], [0.5]])