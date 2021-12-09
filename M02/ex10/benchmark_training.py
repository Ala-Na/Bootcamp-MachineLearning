import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from mylinearregression import MyLinearRegression as MyLR

# Utils functions for files management

def extract_datas(filename):
    assert os.path.isfile(filename)
    datas = pd.read_csv(filename)
    x = np.asarray(datas)
    return x[:, 1:x.shape[1] - 1].reshape(-1, 3), x[:, x.shape[1] - 1:].reshape(-1, 1)

def create_model_file():
    if (os.path.isfile('models.csv')):
        return
    header = ['', 'Form', 'Global MSE', 'Training MSE', 'Testing MSE', 'Testing set', 'Thetas after fit', 'Alpha', 'Max iter']
    f = open('models.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(header)
    f.close()

# Data splitter to obtain testing and training set

def united_shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def data_spliter(x, y, proportion):
    if not isinstance(x, np.ndarray) or x.ndim != 2:
        return None
    if not isinstance(y, np.ndarray) or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(proportion, float) or proportion > 1 or proportion < 0:
        return None
    ind_split = (int)(x.shape[0] * proportion)
    x, y = united_shuffle(x, y)
    return (x[:ind_split, :], x[ind_split:, :], y[:ind_split, :], y[ind_split:, :])

# Function to obtains polynomial form

def add_polynomial_features(x, power):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    if not isinstance(power, int) or power <= 0:
        return None
    X = x
    for i in range(1, power):
        X = np.append(X, ((X[:,0] ** (i + 1)).reshape(-1, 1)), axis=1)
    return X

def get_poly_forms(x_training, x_testing, max_power):
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

# Write model in file

def add_model_to_file(name, global_mse, training_mse, testing_mse, x_testing, thetas, alpha, max_iter):
    df = pd.DataFrame()
    df = df.append({'Form' : name,'Global MSE' : global_mse, 'Training MSE' : training_mse, 'Testing MSE' : testing_mse, \
        'Testing set' : x_testing, 'Thetas after fit' : thetas[:, 0].tolist(), 'Alpha' : alpha, 'Max iter' : max_iter}, ignore_index=True)
    df.to_csv('models.csv', mode='a', header=False)

# Check if model already tried or if tried but current dataset is better

def check_if_model_exist_in_file(form, alpha, max_iter):
    df = pd.read_csv('models.csv')
    find_form = df.loc[df['Form'] == form]
    find_iter = find_form.loc[find_form['Max iter'] == max_iter]
    find_alpha = find_iter.loc[find_iter['Alpha'] == alpha]
    #print(find_alpha.index)
    if find_alpha.empty:
        return True
    return False

def check_if_dataset_is_better(form, x_test, global_mse, training_mse, testing_mse, thetas, alpha, max_iter):
    df = pd.read_csv('models.csv')
    find_form = df.loc[df['Form'] == form]
    find_iter = find_form.loc[find_form['Max iter'] == max_iter]
    find_alpha = find_iter.loc[find_iter['Alpha'] == alpha]
    if find_alpha.empty or find_alpha['Global MSE'].values[0] <= global_mse:
        return False
    #CONTINUE HERE
    # use find_alpha.index ? Like df.drop(index) ?
    

# Training models

def get_alpha(poly_name):
    if (poly_name.find('x2p4') != -1):
        alpha = 3e-29
    elif (poly_name.find('x2p3') != -1 or poly_name.find('x2p2') != -1 or poly_name.find('x1p3') != -1 or poly_name.find('x1p4') != -1):
        alpha = 3e-23
    elif (poly_name.find('x3p4') != -1 or poly_name.find('x1p2') != -1):
        alpha = 3e-11
    else:
        alpha = 3e-8
    return alpha

def train_all_models(base_datas, x_training_forms, x_testing_forms, poly_names):
    y_hat = []
    y_training = base_datas[2]
    y_testing = base_datas[3]
    for i in range(0, len(x_training_forms)):
        # Thetas is initialized to a list of 1
        thetas = [1] * (x_training_forms[i].shape[1] + 1)
        # We try to obtain alpha adapted to each form
        alpha = get_alpha(poly_names[i])
        max_iter = 100
        # First, we check if the model is already present in models.csv or not
        if (check_if_model_exist_in_file(poly_names[i], alpha, max_iter) == False):
            continue
        # We train the model
        form_lr = MyLR(thetas, alpha=alpha, max_iter=max_iter)
        form_lr.fit_(x_training_forms[i], y_training)
        # Obtaining y_hat with training set and corresponding MSE
        y_hat_training = form_lr.predict_(x_training_forms[i])
        training_mse = form_lr.mse_(y_training, y_hat_training)
        if np.isnan(training_mse):
            continue
        # Obtaining y_hat for testing set only and corresponding MSE
        y_hat_testing = form_lr.predict_(x_testing_forms[i])
        testing_mse = form_lr.mse_(y_testing, y_hat_testing)
        if np.isnan(testing_mse):
            continue
        # We seek the global MSE
        x_global = np.concatenate([x_training_forms[i], x_testing_forms[i]])
        y_global = np.concatenate([y_training, y_testing])
        y_hat_global = form_lr.predict_(x_global)
        global_mse = form_lr.mse_(y_global, y_hat_global)
        if np.isnan(global_mse):
            continue
        # We add the model to file
        add_model_to_file(poly_names[i], global_mse, training_mse, testing_mse, x_testing_forms[i], form_lr.thetas, alpha, max_iter)


if __name__ == '__main__':
    x, y = extract_datas('space_avocado.csv')
    create_model_file()
    splited_datas = data_spliter(x, y, 0.8)
    poly_names, x_training_forms, x_testing_forms = get_poly_forms(splited_datas[0], splited_datas[1], 4)
    train_all_models(splited_datas, x_training_forms, x_testing_forms, poly_names)
