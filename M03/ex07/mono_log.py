import numpy as np
import pandas as pd
import os
from data_spliter import data_spliter
import argparse
from my_logistic_regression import MyLogisticRegression as MyLR

def extract_datas(filename):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        datas_array = np.asarray(datas)
        return datas_array
    except:
        print('Something went wrong with extract_datas function')
        return None

def get_splitted_datas(zipcode):
	# Retrieve datas from files
	x = extract_datas('solar_system_census.csv')[:, 1:]
	y = (extract_datas('solar_system_census_planets.csv'))[:, 1].reshape(-1, 1)
	# Check for correct shapes
	if y.shape != (x.shape[0], 1):
		print("Error in shapes !")
		exit(1)
	# Replace zipcode inside y by 1 (is expected zipcode) and 0 (is another zipcode)
	y = np.select([y == zipcode, y != zipcode], [1, 0], y)
	# Split data set in training and testing sets
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
	return x_train, x_test, y_train, y_test	

def train_mono_model(zipcode):
	# Set model
	mono_lor = MyLR(np.array([[1.0], [1.0], [1.0], [1.0]]), alpha=0.0001, max_iter=10000)
	
	# Get datas
	x_train, x_test, y_train, y_test = get_splitted_datas(zipcode)

	# Train model
	print("Training model... (may take some time)")
	mono_lor.fit_(x_train, y_train)
	print("Model trained !\n")

	# Calculate loss and predictions
	x_fullset = np.concatenate([x_train, x_test])
	y_fullset = np.concatenate([y_train, y_test])
	predictions = np.round(mono_lor.predict_(x_fullset))
	loss = mono_lor.loss_(y_fullset, predictions)
	print("Thetas: \033[34m{}\033[0m".format(mono_lor.theta.reshape(1, -1)[0]))
	print("Loss: \033[34m{}\033[0m".format(loss))
	print("Proportion of correct predicted values: \033[34m{}%\033[0m".format((np.sum(predictions == y_fullset) / y_fullset.size) * 100))
	expected = (predictions == y_fullset)
	# Display predictions


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Monoclass Logistic Regression - Solar system")
	parser.add_argument("zipcode", type=int, nargs=1, help="Zipcode of a planet, one single integer between 0 and 3 included (0 = Venus, 1 = Earth, 2 = Mars, 3 = Belt)")
	args = parser.parse_args()
	if args.zipcode[0] > 3 or args.zipcode[0] < 0:
		parser.print_help()
		exit(1)
	train_mono_model(args.zipcode[0])

