import numpy as np
import pandas as pd
import os
from data_spliter import data_spliter
import argparse
from my_logistic_regression import MyLogisticRegression as MyLR
import itertools
import matplotlib.pyplot as plt

def extract_datas(filename):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        datas_array = np.asarray(datas)
        return datas, datas_array
    except:
        print('Something went wrong with extract_datas function')
        return None

def get_splitted_datas(zipcode=None):
	# Retrieve datas from files
	x_datas, x = extract_datas('solar_system_census.csv')
	x = x[:, 1:]
	y_datas, y = extract_datas('solar_system_census_planets.csv')
	y = y[:, 1].reshape(-1, 1)
	# Check for correct shapes
	if y.shape != (x.shape[0], 1):
		print("Error in shapes !")
		exit(1)
	# Replace zipcode inside y by 1 (is expected zipcode) and 0 (is another zipcode)
	if zipcode != None:
		y = np.select([y == zipcode, y != zipcode], [1, 0], y)
	# Split data set in training and testing sets
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
	return x_datas, x, y, x_train, x_test, y_train, y_test


def draw_sub_spots(ax, x, y, label, color, size):
	# Scatter spots on x and y coordinates
    try:
        ax.scatter(x, y, size, label=label, color=color, zorder=2)
    except:
        print('Something went wrong in draw_sub_spots function')

def get_mono_sub_graph(ax, x, y_hat, labels, axis_features, colors):
	try:
		# Draw other planets spots
		draw_sub_spots(ax, x[axis_features[0]].loc[(y_hat == 1)], x[axis_features[1]].loc[(y_hat == 1)], labels[0], colors[0], 25)
		# Draw current planet spots
		draw_sub_spots(ax, x[axis_features[0]].loc[(y_hat == 0)], x[axis_features[1]].loc[(y_hat == 0)], labels[1], colors[1], 15)
		# Set features legends
		ax.set_xlabel(axis_features[0])
		ax.set_ylabel(axis_features[1])
		ax.legend()
	except:
		print('Something went wrong in get_sub_graph function')

def draw_mono_scatter_plots(zipcode, x, y, y_hat):
	# Create pair of datas
	planets = ['Venus', 'Earth', 'Mars', 'Belt']
	planets_colors = ['green', 'blue', 'red', 'yellow']
	labels_t = [planets[zipcode] + ' (true)', 'Other planets']
	labels_p = [planets[zipcode] + ' (pred)', 'Other planets']
	colors = [planets_colors[zipcode], 'grey']
	fig, ax = plt.subplots(2, 3)
	for i, pair in enumerate(itertools.combinations(['weight','height','bone_density'], 2)):
		get_mono_sub_graph(ax[0, i], x, y_hat, labels_p, [pair[0], pair[1]], colors)
		get_mono_sub_graph(ax[1, i], x, y, labels_t, [pair[0], pair[1]], colors)
	plt.show()


def train_mono_model(zipcode, x_original, x_train, x_test, y_train, y_test):
	# Set model
	# Recommend: alpha=0.0001, max_iter=500000
	mono_lor = MyLR(np.array([[1.0], [1.0], [1.0], [1.0]]), alpha=0.0001, max_iter=500000)

	# Train model
	print("Training model for planet {}... (may take some time)".format(zipcode))
	mono_lor.fit_(x_train, y_train)
	print("Model trained !\n")

	# Calculate loss and predictions in train/test/global set
	x_fullset = np.concatenate([x_train, x_test])
	y_fullset = np.concatenate([y_train, y_test])
	print("Thetas: \033[34m{}\033[0m".format(mono_lor.theta.reshape(1, -1)[0]))

	print("Proportion of correct predicted values:")

	y_hat_train = mono_lor.predict_(x_train)
	y_hat_rounded = np.round(y_hat_train)
	correctness = (np.sum(y_hat_rounded == y_train) / y_train.size)
	print("Train set: \033[34m{}%\033[0m".format(correctness * 100))

	y_hat_test = mono_lor.predict_(x_test)
	y_hat_rounded = np.round(y_hat_test)
	y_hat_rounded = np.round(mono_lor.predict_(x_test))
	correctness = (np.sum(y_hat_rounded == y_test) / y_test.size)
	print("Test set: \033[34m{}%\033[0m".format(correctness * 100))

	y_hat_rounded = np.round(mono_lor.predict_(x_fullset))
	correctness = (np.sum(y_hat_rounded == y_fullset) / y_fullset.size)
	print("Global set: \033[34m{}%\033[0m".format(correctness * 100))

	loss = mono_lor.loss_(y_fullset, y_hat_rounded)
	print("Loss on global set: \033[34m{}\033[0m".format(loss))


	# Draw scatter plots for each pair of features
	y_hat_global = mono_lor.predict_(x_original)
	y_hat_rounded = np.round(y_hat_global)
	return y_hat_train, y_hat_test, y_hat_global, y_hat_rounded


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Monoclass Logistic Regression - Solar system")
	parser.add_argument("zipcode", type=int, nargs=1, help="Zipcode of a planet, one single integer between 0 and 3 included (0 = Venus, 1 = Earth, 2 = Mars, 3 = Belt)")
	args = parser.parse_args()
	if args.zipcode[0] > 3 or args.zipcode[0] < 0:
		parser.print_help()
		exit(1)

	print('\033[92mMONOCLASS LOGISTIC REGRESSION\033[0m')
	print('\033[92m------------------------------\033[0m')

	# Get datas
	# We keep x_datas (pandas) and x_original (before split) to draw scatter plots later
	x_datas, x_original, y_original, x_train, x_test, y_train, y_test = get_splitted_datas(args.zipcode[0])

	# Train model
	y_hat_train, y_hat_test, y_hat_global, y_hat_rounded = train_mono_model(args.zipcode[0], x_original, x_train, x_test, y_train, y_test)

	# Draw model
	draw_mono_scatter_plots(args.zipcode[0], x_datas, y_original, y_hat_rounded)
