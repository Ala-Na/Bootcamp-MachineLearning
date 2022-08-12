from benchmark_train import *
from ridge import *
from polynomial_model_extended import add_polynomial_features
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def draw_models_loss():
	try:
		df = pd.read_csv('./models.csv')
		df['Info'] = df['Form'].astype(str) + '\nλ=' + df['Lambda'].astype(str)
		plt.rc('xtick', labelsize=6)
		plt.figure(figsize=(10, 7))
		plt.title('Evaluation curve of loss for differents tested models on validation set')
		plt.plot(df['Info'], df['Loss on validation set'])
		plt.xlabel('Model polynomial form and lambda value')
		plt.ylabel('Loss')
		plt.grid()
		plt.show()
	except:
		print('Something went wrong with trace_evaluation_curve function')

def draw_models_mse():
	try:
		df = pd.read_csv('./models.csv')
		df['Info'] = df['Form'].astype(str) + '\nλ=' + df['Lambda'].astype(str)
		plt.rc('xtick', labelsize=6)
		plt.figure(figsize=(10, 7))
		plt.title('Evaluation curve of mse for differents tested models on validation set')
		plt.plot(df['Info'], df['MSE on validation set'])
		plt.xlabel('Model polynomial form and lambda value')
		plt.ylabel('MSE')
		plt.grid()
		plt.show()
	except:
		print('Something went wrong with trace_evaluation_curve function')

def train_best_model(best, form, x_train, y_train, x_test, y_test):

	x_train = add_polynomial_features(x_train, form)
	x_test = add_polynomial_features(x_test, form)
	thetas = (best['Thetas after fit'].values)[0].strip('[]').replace('\' ', '').split(',')
	thetas = [float(i) for i in thetas]


	print('Training best model with different lambda...\n')

	lambdas = np.linspace(0, 1, 5)
	models = []
	for lambda_ in lambdas:
		ridge = MyRidge(thetas, lambda_=lambda_, alpha=best['Alpha'].values[0], max_iter=50000)
		ridge.fit_(x_train, y_train)
		print('Loss on testing set for λ={}: {:.2f}'.format(lambda_, ridge.loss_(y_test, ridge.predict_(x_test))))
		models.append(ridge)

	return models

def get_continuous_x(x, form):
    try:
        x1_cont = np.linspace(x[:, 0].min(), x[:, 0].max(), 10000)
        x2_cont = np.linspace(x[:, 1].min(), x[:, 1].max(), 10000)
        x3_cont = np.linspace(x[:, 2].min(), x[:, 2].max(), 10000)
        return np.c_[x1_cont, x2_cont, x3_cont]
    except:
        print('Something went wrong with get_poly_continuous_x function')
        return None

def draw_best_model(models, form, x, y):
	colors = ['red', 'lime', 'blue', 'yellow', 'orange']
	features = ['Weight', 'Production distance', 'Time delivery']

	# Scatter plots
	x_poly = mean_normalization(x)
	x_poly = add_polynomial_features(x_poly, form)
	y_hats = []
	lambdas = []
	for model in models:
		y_hat = model.predict_(x_poly)
		y_hats.append(y_hat)
		lambdas.append(model.lambda_)

	fig = plt.figure(figsize=(10*3, 10))
	for i in range(3):
		ax = fig.add_subplot(1, 3, i+1)
		ax.scatter(x[:, i], y, color='seagreen', label='True price', alpha=0.7)
		for k in range(len(lambdas)):
			ax.scatter(x[:, i], y_hats[k], color=colors[k], label='Predicted price for λ={}'.format(lambdas[k]), alpha=0.2)
		ax.set_xlabel(features[i])
		ax.set_ylabel('Price')
		if i == 1:
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
	plt.show()

	# 3D plots
	x_cont = get_continuous_x(x, form)
	x_cont_poly = mean_normalization(x_cont)
	x_cont_poly = add_polynomial_features(x_cont_poly, form)
	y_hats = []
	lambdas = []
	for model in models:
		y_hat = model.predict_(x_cont_poly)
		y_hats.append(y_hat)
		lambdas.append(model.lambda_)

	fig = plt.figure(figsize=(10*3, 10))
	for i in range(3):
		ax = fig.add_subplot(1, 3, i+1, projection='3d')
		j = i + 1 if i < 2 else 0
		ax.scatter3D(x[:, i].flatten(), x[:, j].flatten(), y.flatten(), color='seagreen', label='True price')
		for k in range(len(lambdas)):
			ax.plot3D(x_cont[:, i].flatten(), x_cont[:, j].flatten(), y_hats[k].flatten(), color=colors[k], label='Predicted price for λ={}'.format(lambdas[k]))
		ax.set_xlabel(features[i])
		ax.set_ylabel(features[j])
		ax.set_zlabel('Price')
		if i == 1:
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
	plt.show()

if __name__ == '__main__':
	# Recuperate best model and testing set
	if os.path.isfile('./models.csv') and os.path.isfile('./sets.npz'):
		print('Models were already trained !')
		best = find_best_model('./models.csv')
		sets = np.load('./sets.npz')
		x, y, x_train, y_train, x_test, y_test = sets['x'], sets['y'], sets['x_train'], sets['y_train'], sets['x_test'], sets['y_test']
	else:
		best, x, y, x_train, y_train, x_test, y_test = launch_benchmark(50000)

	# Recuperate polynomial form
	forms = ['1', '2', '3' , '4']
	form = 0
	for i in range(len(forms)):
		if (forms[i] in best['Form'].values[0]):
			form = i + 1

	# Draw evaluation curve for models
	draw_models_loss()
	draw_models_mse()

	# Train best model
	models = train_best_model(best, form, x_train, y_train, x_test, y_test)

	# Draw best model
	draw_best_model(models, form, x, y)
