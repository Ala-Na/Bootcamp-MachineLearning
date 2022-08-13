from benchmark_train import *
from my_logistic_regression import MyLogisticRegression as MyLoR
from data_spliter import data_spliter
from polynomial_model_extended import add_polynomial_features
from other_metrics import *
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import itertools

def draw_evaluation_bar_plots():
	try:
		df = pd.read_csv('./models.csv')
		df['Info'] = df['Form'].astype(str) + '\nλ=' + df['Lambda'].astype(str)
		plt.rc('xtick', labelsize=6)
		plt.figure(figsize=(10, 7))
		plt.title('F1 score for differents models and lambda')
		plt.bar(df['Info'], df['F1 score'])
		plt.xlabel('Model polynomial form and lambda value')
		plt.ylabel('F1 score')
		plt.grid()
		plt.show()
	except:
		print('Something went wrong with trace_evaluation_curve function')

def generate_others_polynomial_forms(max_degree, to_avoid, x, x_train, x_test):
	forms = []
	x_poly = []
	x_train_poly = []
	x_test_poly = []
	for i in range(1, max_degree + 1):
		if i != to_avoid:
			forms.append('x^' + str(i))
			x_poly.append(add_polynomial_features(x, i))
			x_train_poly.append(add_polynomial_features(x_train, i))
			x_test_poly.append(add_polynomial_features(x_test, i))
	return forms, x_poly, x_train_poly, x_test_poly

def draw_comparison_scatter_plot(x, y, y_hat):
	labels_pred = ['Venus citizen (prediction)', 'Earth citizen (prediction)', 'Mars citizen (prediction)', 'Belt citizen (prediction)']
	labels_true = ['Venus citizen (true)', 'Earth citizen (true)', 'Mars citizen (true)', 'Belt citizen (true)']
	colors = ['green', 'blue', 'red', 'yellow']
	fig = plt.figure(figsize=(10*3, 10))
	# Create pair of datas and draw plot accordingly
	for i, pair in enumerate(itertools.combinations(['weight','height','bone_density'], 2)):
		ax = fig.add_subplot(1, 3, i + 1)
		# Draw Venus spots
		ax.scatter(x[pair[0]].loc[(y_hat == 0)], x[pair[1]].loc[(y_hat == 0)], 50, label=labels_pred[0], color=colors[0], zorder=1)
		ax.scatter(x[pair[0]].loc[(y == 0)], x[pair[1]].loc[(y == 0)], 50, label=labels_true[0], facecolors='none', edgecolors=colors[0], zorder=2)
		# Draw Earth spots
		ax.scatter(x[pair[0]].loc[(y_hat == 1)], x[pair[1]].loc[(y_hat == 1)], 50, label=labels_pred[1], color=colors[1], zorder=1)
		ax.scatter(x[pair[0]].loc[(y == 1)], x[pair[1]].loc[(y == 1)], 50, label=labels_true[1], facecolors='none', edgecolors=colors[1], zorder=2)
		# Draw Mars spots
		ax.scatter(x[pair[0]].loc[(y_hat == 2)], x[pair[1]].loc[(y_hat == 2)], 50, label=labels_pred[2], color=colors[2], zorder=1)
		ax.scatter(x[pair[0]].loc[(y == 2)], x[pair[1]].loc[(y == 2)], 50, label=labels_true[2], facecolors='none', edgecolors=colors[2], zorder=2)
		# Draw Belt spots
		ax.scatter(x[pair[0]].loc[(y_hat == 3)], x[pair[1]].loc[(y_hat == 3)], 50, label=labels_pred[3], color=colors[3], zorder=1)
		ax.scatter(x[pair[0]].loc[(y == 3)], x[pair[1]].loc[(y == 3)], 50, label=labels_true[3], facecolors='none', edgecolors=colors[3], zorder=2)
		ax.set_xlabel(pair[0])
		ax.set_ylabel(pair[1])
		if i == 1:
			ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=8)
	plt.show()

if __name__ == '__main__':
	# Recuperate best model
	if os.path.isfile('./models.csv'):
		print('Models were already trained !')
		best_form = find_best_model_form('./models.csv')
	else:
		best_form = launch_benchmark(15000, 0.005)

	# Recuperate polynomial model form
	forms = ['1', '2', '3' , '4']
	form = 0
	for i in range(len(forms)):
		if (forms[i] in best_form):
			form = i + 1

	## Draw bar plot of scores for each models and their lambda values
	draw_evaluation_bar_plots()

	# Recuperate data for training model from scratch
	X_df, X, y_df, y = extract_solar_datas()

	# Value can be too big and screw up calculations : Let's scale it
	x = mean_normalization(X)

	# Split dataset into training/cross-validation and tests sets
	# and get correct polynomial form
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)
	x_train_poly = add_polynomial_features(x_train, form)
	x_test_poly = add_polynomial_features(x_test, form)

	# Train model with different lambdas and save f1_scores
	scores = []
	y_hats = []
	lambdas = np.round(np.linspace(0, 1, 6), 2)
	print('\n\033[92mTraining best model with different lambdas:\033[0m')
	for lambda_ in lambdas:
		score, _, y_hat = train_model(30000, form, add_polynomial_features(x, form), x_train_poly, y_train, x_test_poly, y_test, lambda_, 0.005, on_test=True)
		y_hats.append(y_hat)
		scores.append(score)

	# Comparison plot between predicted and true values for best model
	max_score = max(scores)
	max_index = scores.index(max_score)
	y_hat = y_hats[max_index]
	draw_comparison_scatter_plot(X_df, y, y_hat)
