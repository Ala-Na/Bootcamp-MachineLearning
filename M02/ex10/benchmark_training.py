import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

# Utils functions for files management

def extract_datas(filename):
    assert os.path.isfile(filename)
    datas = pd.read_csv(filename)
    x = np.asarray(datas)
    return x[:, 1:x.shape[1] - 1].reshape(-1, 3), x[:, x.shape[1] - 1:].reshape(-1, 1)

def create_model_file():
    if (os.path.isfile('models.csv')):
        f = open('models.csv', 'a+')
        return f
    header = ['Form', 'Global MSE', 'Training MSE', 'Testing MSE', 'Training set', 'Testing set', 'Thetas', 'Alpha', 'Max iter']
    f = open('models.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(header)
    return f

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

def train_all_models(x_training_forms, x_testing_forms, y_training, y_testing, poly_names, file):
    y_hat = []
    for i in range(0, len(x_training_forms)):
        thetas = [1] * (x_training_forms[i].shape[1] + 1)
        alpha = get_alpha(poly_names[i])
        print(alpha)
        
        form_lr = MyLR(thetas, alpha=alpha, max_iter=1000)
        form_lr.fit_(x_training_forms[i], y_training)

        # lines to show thetas
        np.set_printoptions(precision=2)
        list = form_lr.thetas[:,0]
        print(poly_names[i], ": ", list)
        #end lines to show thetas

        y_hat_training = form_lr.predict_(x_training_forms[i])
        training_mse = form_lr.mse_(y_training, y_hat_training)
        print("training mse : ", training_mse)

        y_hat_testing = form_lr.predict_(x_testing_forms[i])
        testing_mse = form_lr.mse_(y_testing, y_hat_testing)
        print("testing mse : ", testing_mse)

        x_global = np.concatenate([x_training_forms[i], x_testing_forms[i]])
        y_global = np.concatenate([y_training, y_testing])
        y_hat_global = form_lr.predict_(x_global)
        global_mse = form_lr.mse_(y_global, y_hat_global)
        print("global mse : ", global_mse)


if __name__ == '__main__':
    x, y = extract_datas('space_avocado.csv')
    file = create_model_file()
    splited_datas = data_spliter(x, y, 0.8)
    x_base_training = splited_datas[0]
    y_training = splited_datas[2]
    x_base_testing = splited_datas[1]
    y_testing = splited_datas[3]
    poly_names, x_training_forms, x_testing_forms = get_poly_forms(x_base_training, x_base_testing, 4)
    train_all_models(x_training_forms, x_testing_forms, y_training, y_testing, poly_names, file)
