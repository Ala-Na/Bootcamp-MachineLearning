from mono_log import *

def get_multi_sub_graph(ax, x, y_hat, labels, axis_features, colors):
	try:
		# Draw Venus spots
		draw_sub_spots(ax, x[axis_features[0]].loc[(y_hat == 0)], x[axis_features[1]].loc[(y_hat == 0)], labels[0], colors[0], 15)
		# Draw Earth spots
		draw_sub_spots(ax, x[axis_features[0]].loc[(y_hat == 1)], x[axis_features[1]].loc[(y_hat == 1)], labels[1], colors[1], 15)
		# Draw Mars spots
		draw_sub_spots(ax, x[axis_features[0]].loc[(y_hat == 2)], x[axis_features[1]].loc[(y_hat == 2)], labels[2], colors[2], 15)
		# Draw Belt spots
		draw_sub_spots(ax, x[axis_features[0]].loc[(y_hat == 3)], x[axis_features[1]].loc[(y_hat == 3)], labels[3], colors[3], 15)
		# Set features legends
		ax.set_xlabel(axis_features[0])
		ax.set_ylabel(axis_features[1])
		ax.legend()
	except:
		print('Something went wrong in get_sub_graph function')

def draw_multi_scatter_plots(x, y, y_hat):
	planets_p = ['Venus (pred)', 'Earth (pred)', 'Mars (pred)', 'Belt (pred)']
	planets_t = ['Venus (true)', 'Earth (true)', 'Mars (true)', 'Belt (true)']
	planets_colors = ['green', 'blue', 'red', 'yellow']
	fig, ax = plt.subplots(2, 3)
	# Create pair of datas and draw plot accordingly
	for i, pair in enumerate(itertools.combinations(['weight','height','bone_density'], 2)):
		get_multi_sub_graph(ax[0, i], x, y_hat, planets_p, [pair[0], pair[1]], planets_colors)
		get_multi_sub_graph(ax[1, i], x, y, planets_t, [pair[0], pair[1]], planets_colors)
	plt.show()

def train_multi_model(x_original, y_original, x_train, x_test, y_train, y_test):
	planets_y_hat_train = []
	planets_y_hat_test = []
	planets_y_hat_global = []

	print('\n\033[92mFirst step: Perform monoclass logitic regression for each planet\033[0m\n')

	for planet_idx in range(0, 4):
		print("--------------------------")
		y_train_planet = np.select([y_train == planet_idx, y_train != planet_idx], [1, 0], y_train)
		y_test_planet = np.select([y_test == planet_idx, y_test != planet_idx], [1, 0], y_test)
		y_hat_train, y_hat_test, y_hat_global, y_hat_rounded = train_mono_model(planet_idx, x_original, x_train, x_test, y_train_planet, y_test_planet)
		planets_y_hat_train.append(y_hat_train)
		planets_y_hat_test.append(y_hat_test)
		planets_y_hat_global.append(y_hat_global)
		print("--------------------------")
		print('\n')

	print('\033[92mSecond step: Deduce planet classification for each citizen\033[0m\n')

	planets_predictions_train = np.argmax(planets_y_hat_train, axis=0).reshape(-1, 1)
	planets_predictions_test = np.argmax(planets_y_hat_test, axis=0).reshape(-1, 1)
	planets_predictions_global = np.argmax(planets_y_hat_global, axis=0).reshape(-1, 1)

	print("Proportion of correct predicted values:")

	correctness = (np.sum(planets_predictions_train == y_train) / y_train.size)
	print("Train set: \033[34m{}%\033[0m".format(correctness * 100))

	correctness = (np.sum(planets_predictions_test == y_test) / y_test.size)
	print("Test set: \033[34m{}%\033[0m".format(correctness * 100))

	correctness = (np.sum(planets_predictions_global == y_original) / y_original.size)
	print("Global set: \033[34m{}%\033[0m".format(correctness * 100))

	return planets_predictions_global

if __name__ == '__main__':
	print('\033[92mMULTICLASS LOGISTIC REGRESSION\033[0m')
	print('\033[92m------------------------------\033[0m')

	# Get datas
	# We keep x_datas (pandas) and x_original (before split) to draw scatter plots later
	x_datas, x_original, y_original, x_train, x_test, y_train, y_test = get_splitted_datas()

	# Train model
	y_hat = train_multi_model(x_original, y_original, x_train, x_test, y_train, y_test)

	# Draw scatter plots
	draw_multi_scatter_plots(x_datas, y_original, y_hat)
