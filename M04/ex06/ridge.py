

class MyRidge(ParentClass):
"""
Description:
My personnal ridge regression class to fit like a boss.
"""
def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
self.alpha = alpha
self.max_iter = max_iter
self.thetas = thetas
self.lambda_ = lambda_
... Your code here ...
... other methods ...
