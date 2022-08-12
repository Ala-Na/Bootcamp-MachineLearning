from my_logistic_regression import MyLogisticRegression as MyLoR
from data_spliter import data_spliter
from polynomial_model_extended import add_polynomial_features
from other_metrics import *
import numpy as np
import os
import csv
import pandas as pd

# File management functions

def extract_datas(filename):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        datas_array = np.asarray(datas)
        return datas, datas_array
    except:
        print('Something went wrong with extract_datas function')
        return None

def create_model_file():
	try:
		if (os.path.isfile('models.csv')):
			os.remove('models.csv')
		header = ['Form', 'Lambda', 'F1 score', 'Alpha']
		f = open('models.csv', 'w')
		writer = csv.writer(f)
		writer.writerow(header)
		f.close()
	except:
		print('Something went wrong with create_model_file function')

def extract_solar_datas(zipcode=None):
	# Retrieve datas from files
	x_df, x = extract_datas('solar_system_census.csv')
	x = x[:, 1:]
	y_df, y = extract_datas('solar_system_census_planets.csv')
	y = y[:, 1].reshape(-1, 1)
	# Check for correct shapes
	if y.shape != (x.shape[0], 1):
		print("Error in shapes !")
		exit(1)
	return x_df, x, y_df, y

def add_model_to_file(name, f1_score, alpha, lambda_):
	df = pd.DataFrame([{'Form' : name,  'Lambda': lambda_, 'F1 score' : f1_score, \
			'Alpha' : alpha}])
	df.to_csv('models.csv', mode='a', header=False, index=False)

# Function to normalize values and avoid calculus overflow

def mean_normalization(x):
	mean = np.mean(x, axis=0)
	std = np.std(x, axis=0)
	x = (x - mean) / std
	return x

# Function to replace zipcode in a one-vs-all logic inside y (results)
# 1 = expected zipcode, 0 = other zipcode
def replace_zipcode(y, zipcode):
	y = np.select([y == zipcode, y != zipcode], [1, 0], y)
	return y

# Function to generate polynomial forms
def generate_polynomial_forms(max_degree, x, x_valid):
	forms = []
	x_poly = []
	x_val_poly = []
	for i in range(1, max_degree + 1):
		forms.append('x^' + str(i))
		x_poly.append(add_polynomial_features(x, i))
		x_val_poly.append(add_polynomial_features(x_valid, i))
	return forms, x_poly, x_val_poly

# Function to train model in one vs all
def one_vs_all(x, y, x_val, thetas, max_iter, alpha, lambda_):
	validation_y_hat = []
	global_y_hat = []
	global_x = np.vstack((x, x_val))
	for zipcode in range (0, 4):
		y_tmp = replace_zipcode(y, zipcode)
		submodel = MyLoR(thetas, alpha=alpha, lambda_=lambda_, max_iter=max_iter)
		submodel.fit_(x, y_tmp)
		validation_y_hat.append(submodel.predict_(x_val))
		global_y_hat.append(submodel.predict_(global_x))
	model_y_hat = np.argmax(validation_y_hat, axis=0).reshape(-1, 1)
	model_global_y_hat = np.argmax(global_y_hat, axis=0).reshape(-1, 1)
	return model_y_hat, model_global_y_hat

# Function to train models

def train_model(max_iter, form, x, y, x_val, y_val, lambda_, alpha, on_test):
	print('\033[33mTraining model of form {} with λ={}...\033[0m'.format(form, lambda_))
	thetas = [1] * (x.shape[1] + 1)
	# Train model in one vs all logic
	y_hat, global_y_hat = one_vs_all(x, y, x_val, thetas, max_iter, alpha, lambda_)
	# Evaluate model on validation set
	f1_score = f1_score_(y_val, y_hat)
	if not on_test:
		print('F1 score on validation set: {:.2f}\n'.format(f1_score))
	else:
		print('F1 score on testing set: {:.2f}\n'.format(f1_score))
	# Save model
	if not on_test:
		add_model_to_file(form, f1_score, alpha, lambda_)
	return f1_score, y_hat, global_y_hat

# Function to find best model (f1 score closest to 1)

def find_best_model_form(filename):
	assert os.path.isfile(filename)
	df = pd.read_csv(filename)
	mean_f1_score = df.groupby('Form')['F1 score'].mean()
	best_form = mean_f1_score.idxmax()
	print('\033[92mBest model (according to F1 score):\033[0m Form \033[34m{}\033[0m'.format(best_form))
	return best_form

def launch_benchmark(max_iter, alpha, on_test=False):
	# Create model file and extract data as array
	create_model_file()
	X_df, X, y_df, y = extract_solar_datas()

	# Value can be too big and screw up calculations : Let's scale it
	x = mean_normalization(X)

	if not on_test:
	# Split dataset into training/cross-validation and tests sets
		x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)
		x_valid, x_test, y_valid, y_test = data_spliter(x_test, y_test, 0.5)
	else:
		x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)
		x_valid, y_valid = x_test, y_test


	# Generate polynomial forms with maximum degree of 3
	forms, x_poly, x_val_poly = generate_polynomial_forms(3, x_train, x_valid)
	
	# Train differents models with differents lambdas
	lambdas = np.round(np.linspace(0, 1, 6), 2)
	for form in range(len(forms)):
		for lambda_ in lambdas:
			train_model(max_iter, forms[form], x_poly[form], y_train, x_val_poly[form], y_valid, lambda_, alpha, on_test)
	
	# Display best model found
	best_form = find_best_model_form('./models.csv')
	return best_form

if __name__ == '__main__':
	launch_benchmark(60000, 0.005)
