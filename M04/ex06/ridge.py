from mylinearregression import MyLinearRegression as MyLR

class MyRidge(MyLR):
	"""
	Description:
		My personnal ridge regression class to fit like a boss.
	"""
	
	def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=0.5):
		assert isinstance(lambda_, float) and lambda_ >= 0
		super(MyRidge, self).__init__(thetas, alpha, max_iter)
		self.lambda_ = lambda_
