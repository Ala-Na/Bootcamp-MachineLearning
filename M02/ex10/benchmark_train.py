import numpy as np
import os
import csv
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
from data_spliter import data_spliter
from polynomial_model import add_polynomial_features

# Utils functions for files management

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
	try:
		if (os.path.isfile('models.csv')):
			os.remove('models.csv')
		header = ['Form', 'Loss on testing set', 'MSE on testing set', 'Thetas after fit', 'Alpha']
		f = open('models.csv', 'w')
		writer = csv.writer(f)
		writer.writerow(header)
		f.close()
	except:
		print('Something went wrong with create_model_file function')

# Function to normalize data and avoid calculus overflow

def mean_normalization(x):
	mean = np.mean(x, axis=0)
	std = np.std(x, axis=0)
	x = (x - mean) / std
	return x

# Function to get crossed and polynomial forms

def get_crossed_form(x_training, x_testing):
    try:
        crossed_form = ['x1*x2 x3', 'x1*x3 x2', 'x1 x2*x3', 'x1*x2*x3']
        x_train = []
        x_train.append(np.c_[np.multiply(x_training[:, 0], x_training[:, 1]), x_training[:, 2]])
        x_train.append(np.c_[np.multiply(x_training[:, 0], x_training[:, 2]), x_training[:, 1]])
        x_train.append(np.c_[x_training[:, 0], np.multiply(x_training[:, 1], x_training[:, 2])])
        x_train.append(np.multiply(np.multiply(x_training[:, 0], x_training[:, 1]), x_training[:, 2]).reshape(-1, 1))
        x_test = []
        x_test.append(np.c_[np.multiply(x_testing[:, 0], x_testing[:, 1]), x_testing[:, 2]])
        x_test.append(np.c_[np.multiply(x_testing[:, 0], x_testing[:, 2]), x_testing[:, 1]])
        x_test.append(np.c_[x_testing[:, 0], np.multiply(x_testing[:, 1], x_testing[:, 2])])
        x_test.append(np.multiply(np.multiply(x_testing[:, 0], x_testing[:, 1]), x_testing[:, 2]).reshape(-1, 1))
        return crossed_form, x_train, x_test
    except:
        print('Something went wrong with get_crossed_form function')
        return None

def get_poly_forms(x_training, x_testing, max_power):
    try:
        poly_name = []
        x_training_poly = []
        x_testing_poly = []
        x1_train = add_polynomial_features(x_training[:, 0].reshape(-1, 1), max_power)
        x2_train = add_polynomial_features(x_training[:, 1].reshape(-1, 1), max_power)
        x3_train = add_polynomial_features(x_training[:, 2].reshape(-1, 1), max_power)
        x1_test = add_polynomial_features(x_testing[:, 0].reshape(-1, 1), max_power)
        x2_test = add_polynomial_features(x_testing[:, 1].reshape(-1, 1), max_power)
        x3_test = add_polynomial_features(x_testing[:, 2].reshape(-1, 1), max_power)
        for i in range(0, max_power):
            str_i = 'x1p' + str(i + 1)
            for j in range(0, max_power):
                str_j = str_i + ' x2p' + str(j + 1)
                for k in range(0, max_power):
                    str_k = str_j + ' x3p' + str(k + 1)
                    train_array = np.c_[x1_train[:, :i + 1], x2_train[:, :j + 1], x3_train[:, :k + 1]]
                    test_array = np.c_[x1_test[:, :i + 1], x2_test[:, :j + 1], x3_test[:, :k + 1]]
                    poly_name.append(str_k)
                    x_training_poly.append(train_array)
                    x_testing_poly.append(test_array)
        return poly_name, x_training_poly, x_testing_poly
    except:
        print('Something went wrong with get_poly_forms function')
        return None    

# Write model in file

def add_model_to_file(name, loss, mse, thetas, alpha):
    try:
        df = pd.DataFrame({'Form' : name, 'Loss on testing set' : loss, \
			'MSE on testing set': mse, 'Thetas after fit' : [thetas[:, 0].tolist()], 'Alpha' : alpha})
        df.to_csv('models.csv', mode='a', header=False, index=False)
    except:
        print('Something went wrong with add_model_to_file function')

# Training models

def train_all_models(max_iter, x_training_forms, x_testing_forms, poly_names, y_train, y_test):
    try:
        for i in range(0, len(x_training_forms)):

            # Thetas is initialized to a list of 1
            thetas = [1] * (x_training_forms[i].shape[1] + 1)
            alpha = 0.005

            # We train the model
            print("\033[33mTraining model of form {}\033[0m".format(poly_names[i]))
            form_lr = MyLR(np.array(thetas).reshape(-1, 1), alpha=alpha, max_iter=max_iter)
            form_lr.fit_(x_training_forms[i], y_train)

            # We calculate loss and MSE on testing set
            y_hat = form_lr.predict_(x_testing_forms[i])

            loss = form_lr.loss_(y_test, y_hat)
            if np.isnan(loss) or loss == float("inf"):
                print('\033[91mError in calculation (try to reduce alpha)\033[0m')
                continue
            print('Loss on testing set: {:.2f}'.format(loss))

            mse = form_lr.mse_(y_test, y_hat)
            if np.isnan(mse) or mse == float("inf"):
                print('\033[91mError in calculation (try to reduce alpha)\033[0m')
                continue
            print('MSE on testing set:  {:.2f}'.format(mse))

            # We add the model to file
            add_model_to_file(poly_names[i], loss, mse, form_lr.theta, alpha)
            print('\033[92mOK\033[0m\n')
    except:
        print("Something went wrong with train_all_models function")

# Find best model according to MSE

def find_best_model(filename):
    assert os.path.isfile(filename)
    df = pd.read_csv(filename)
    min_col = df.min()
    best = df.loc[df['MSE on testing set'] == min_col['MSE on testing set']]
    print('\033[92mBest model (according to MSE):\033[0m Form \033[34m{}\033[0m of loss \033[34m{:.2f}\033[0m and mse \033[34m{:.2f}\033[0m'.format(best['Form'].values[0], best['Loss on testing set'].values[0], best['MSE on testing set'].values[0]))
    return best

# Save datas sets to use the same in space_avocado.py

def save_sets(x, y, x_train, y_train, x_test, y_test):
	if (os.path.isfile('sets.npz')):
			os.remove('sets.npz')
	try:
		np.savez('sets.npz', x=x, y=y, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
	except:
		print('Something went wrong with save_sets function')
		return None

# Benchmark function

def launch_benchmark(max_iter):
    filename = './space_avocado.csv'

    # Create model file and extract data as array
    create_model_file()
    X, y = extract_datas(filename)

    # Value can be too big and screw up calculations : Let's scale it
    x = mean_normalization(X)

    # Split dataset into training and tests sets
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.8)

    # Generate polynomial forms with maximum degree of 4
    poly_names, x_poly_train, x_poly_test = get_poly_forms(x_train, x_test, 4)
    crossed_names, x_cross_train, x_cross_test = get_crossed_form(x_train, x_test)
    poly_names.extend(crossed_names)
    x_poly_train.extend(x_cross_train)
    x_poly_test.extend(x_cross_test)
    train_all_models(max_iter, x_poly_train, x_poly_test, poly_names, y_train, y_test)

    # Display best model found
    best = find_best_model('./models.csv')

    # Save sets for space_avocado.py
    save_sets(X, y, x_train, y_train, x_test, y_test)
    return best, X, y, x_train, y_train, x_test, y_test

if __name__ == '__main__':
	launch_benchmark(50000)
