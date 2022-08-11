from data_spliter import data_spliter
from ridge import MyRidge
from polynomial_model_extended import add_polynomial_features
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

def extract_datas(filename):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        x = np.asarray(datas)
        return x[:, 1:x.shape[1] - 1].reshape(-1, 3), x[:, x.shape[1] - 1:].reshape(-1, 1)
    except:
        print('Something went wrong with extract_datas function')
        return None

def create_model_file():
    if (os.path.isfile('models.csv')):
        return
    try:
        header = ['Form', 'Lambda', 'Loss on validation set', 'Thetas after fit', 'Alpha']
        f = open('models.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(header)
        f.close()
    except:
        print('Something went wrong with create_model_file function')

def mean_normalization(x):
	mean = np.mean(x)
	std = np.std(x)
	x = (x - mean) / std
	return x

def check_if_dataset_is_better(form, lambda_, loss):
	try:
		df = pd.read_csv('models.csv')
		find_form = df.loc[df['Form'] == form]
		if find_form.empty:
			return True
		find_lambda = find_form.loc[find_form['Lambda'] == lambda_]
		if find_lambda.empty:
			return True
		if np.isnan(find_lambda['Loss on validation set'].values[0]):
			return True
		if loss == float("inf") or find_lambda['Loss on validation set'].values[0] <= loss:
			return False
		idx = (find_lambda.index.tolist())[0]
		df.drop(idx, inplace=True)
		df.to_csv('models.csv', index=False)
		return True
	except:
		print('Something went wrong with check_if_dataset_is_better function')
		return None

def add_model_to_file(name, loss, thetas, alpha, lambda_):
	try:
		df = pd.DataFrame({'Form' : name,  'Lambda': lambda_, 'Loss on validation set' : loss, \
			'Thetas after fit' : [thetas[:, 0].tolist()], 'Alpha' : alpha})
		df.to_csv('models.csv', mode='a', header=False, index=False)
	except:
		print('Something went wrong with add_model_to_file function')

def generate_polynomial_forms(max_degree, x, x_valid):
	forms = []
	x_poly = []
	x_val_poly = []
	for i in range(1, max_degree + 1):
		forms.append('x^' + str(i))
		x_poly.append(add_polynomial_features(x, i))
		x_val_poly.append(add_polynomial_features(x_valid, i))
	return forms, x_poly, x_val_poly

def recuperate_thetas(form, lambda_, len_thetas):
	try:
		df = pd.read_csv('models.csv')
		find_form = df.loc[df['Form'] == form]
		if find_form.empty:
			return [1] * len_thetas
		find_lambda = find_form.loc[find_form['Lambda'] == lambda_]
		if find_lambda.empty:
			return [1] * len_thetas
		thetas = (find_lambda['Thetas after fit'].values)[0].strip('[]').replace('\' ', '').split(',')
		thetas = [float(i) for i in thetas]
		if np.isnan(np.sum(thetas)):
			return [1] * len_thetas
		return thetas
	except:
		print('Something went wrong with recuperate_thetas function')
		return None

def train_model(form, x, y, x_val, y_val, lambda_):
	print('Training model of polynomial form {} with lambda {}...'.format(form, lambda_))

	# Recuperate thetas from previous training
	thetas = recuperate_thetas(form, lambda_, x.shape[1] + 1)

	# Create model
	ridge = MyRidge(thetas, alpha=0.01, lambda_=lambda_, max_iter=10000)
	ridge.fit_(x, y)

	# Evaluate model on validation set
	loss = ridge.loss_(y_val, ridge.predict_(x_val))
	print('Loss on validation set: {}\n'.format(loss))

	# Save model
	if check_if_dataset_is_better(form, lambda_, loss) is True:
		add_model_to_file(form, loss, ridge.theta, ridge.alpha, lambda_)
	return None


if __name__ == '__main__':

	# Create model file and extract data as array
	create_model_file()
	x, y = extract_datas('./space_avocado.csv')

	# Value can be too big and screw up calculations : Let's scale it
	x = mean_normalization(x)

	# Split dataset i nto training/cross-validation and tests sets
	x_train, x_test, y_train, y_test = data_spliter(x, y, 0.6)
	x_valid, x_test, y_valid, y_test = data_spliter(x_test, y_test, 0.5)

	# Generate polynomial forms with maximum degree of 4
	forms, x_poly, x_val_poly = generate_polynomial_forms(4, x_train, x_valid)
	
	# Train differents models with differents lambdas
	lambdas = np.round(np.linspace(0, 1, 6), 2)
	for form in range(len(forms)):
		for lambda_ in lambdas:
			train_model(forms[form], x_poly[form], y_train, x_val_poly[form], y_valid, lambda_)
