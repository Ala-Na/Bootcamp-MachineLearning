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
        header = ['', 'Form', 'Global MSE', 'Thetas after fit', 'Alpha']
        f = open('models.csv', 'w')
        writer = csv.writer(f)
        writer.writerow(header)
        f.close()
    except:
        print('Something went wrong with create_model_file function')

# Data splitter to obtain testing and training set

def united_shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]

def data_spliter(x, y, proportion):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return None
    if not isinstance(y, np.ndarray) or not np.issubdtype(y.dtype, np.number) or y.ndim != 2 or y.shape != (x.shape[0], 1):
        return None
    if not isinstance(proportion, float) or proportion > 1 or proportion < 0:
        return None
    try:
        ind_split = (int)(x.shape[0] * proportion)
        x, y = united_shuffle(x, y)
        return (x[:ind_split, :], x[ind_split:, :], y[:ind_split, :], y[ind_split:, :])
    except:
        return None

# Function to get crossed form

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
    
# Function to obtains polynomial form

def add_polynomial_features(x, power):
    if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.number) or x.ndim != 2 or x.shape[1] != 1 or x.shape[0] == 0:
        return None
    if not isinstance(power, int) or power <= 0:
        return None
    try:
        X = x
        for i in range(1, power):
            X = np.append(X, ((X[:,0] ** (i + 1)).reshape(-1, 1)), axis=1)
        return X
    except:
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

def add_model_to_file(name, global_mse, thetas, alpha):
    try:
        df = pd.DataFrame()
        df = df.append({'Form' : name, 'Global MSE' : global_mse, \
            'Thetas after fit' : thetas[:, 0].tolist(), 'Alpha' : alpha}, ignore_index=True)
        df.to_csv('models.csv', mode='a', header=False)
    except:
        print('Something went wrong with add_model_to_file function')

# Check if model already tried or if tried but current dataset is better

def check_if_dataset_is_better(form, global_mse):
    try:
        df = pd.read_csv('models.csv')
        find_form = df.loc[df['Form'] == form]
        if find_form.empty:
            return True
        if global_mse == float("inf") or find_form['Global MSE'].values[0] <= global_mse:
            return False
        idx = (find_form.index.tolist())[0]
        df.drop(idx, inplace=True)
        df.to_csv('models.csv', index=False)
        return True
    except:
        print('Something went wrong with check_if_dataset_is_better function')
        return None

# Training models

def get_alpha(poly_name):
    try:
        if (poly_name.find('x2p4') != -1):
            alpha = 3e-28
        elif (poly_name.find('x2p3') != -1 or poly_name.find('x2p2') != -1 or poly_name.find('x1p3') != -1 or poly_name.find('x1p4') != -1):
            alpha = 3e-21
        elif (poly_name.find('x3p4') != -1 or poly_name.find('x1*x2') != -1):
            alpha = 3e-20
        elif (poly_name.find('x1p2') != -1 or poly_name.find('*') != -1):
            alpha = 3e-8
        else:
            alpha = 3e-7
        return alpha
    except:
        print('Something went wrong in get_alpha function')
        return None

def recuperate_thetas(form, len_thetas):
    try:
        df = pd.read_csv('models.csv')
        find_form = df.loc[df['Form'] == form]
        if find_form.empty:
            return [1] * len_thetas
        thetas = (find_form['Thetas after fit'].values)[0].strip('[]').replace('\' ', '').split(',')
        thetas = [float(i) for i in thetas]
        return thetas
    except:
        print('Something went wrong with recuperate_thetas function')
        return None

def train_all_models(base_datas, x_training_forms, x_testing_forms, poly_names):
    try:
        y_training = base_datas[2]
        y_testing = base_datas[3]
        for i in range(0, len(x_training_forms)):
            # Thetas is initialized to a list of 1
            thetas = recuperate_thetas(poly_names[i], x_training_forms[i].shape[1] + 1)
            # We try to obtain alpha adapted to each form
            alpha = get_alpha(poly_names[i])
            max_iter = 100
            # We train the model
            print("\033[33mTraining form {} for alpha {} and max_iter {}\033[0m".format(poly_names[i], alpha, max_iter))
            form_lr = MyLR(np.array(thetas).reshape(-1, 1), alpha=alpha, max_iter=max_iter)
            form_lr.fit_(x_training_forms[i], y_training)
            # We seek the global MSE
            x_global = np.concatenate([x_training_forms[i], x_testing_forms[i]])
            y_global = np.concatenate([y_training, y_testing])
            y_hat_global = form_lr.predict_(x_global)
            global_mse = form_lr.mse_(y_global, y_hat_global)
            print(global_mse)
            if np.isnan(global_mse) or global_mse == float("inf"):
                print('\033[91mError in calculation (try to reduce alpha)\033[0m')
                continue
            # We add the model to file
            if check_if_dataset_is_better(poly_names[i], global_mse) is True:
                add_model_to_file(poly_names[i], global_mse, form_lr.theta, alpha)
            print('\033[92mOK\033[0m')
    except:
        print("Something went wrong with train_all_models function")


if __name__ == '__main__':
    try:
        x, y = extract_datas('space_avocado.csv')
        create_model_file()
        splited_datas = data_spliter(x, y, 0.8)
        poly_names, x_training_forms, x_testing_forms = get_poly_forms(splited_datas[0], splited_datas[1], 4)
        crossed_form, x_cross_train, x_cross_test = get_crossed_form(splited_datas[0], splited_datas[1])
        poly_names.extend(crossed_form)
        x_training_forms.extend(x_cross_train)
        x_testing_forms.extend(x_cross_test)
        train_all_models(splited_datas, x_training_forms, x_testing_forms, poly_names)
    except:
        print('Something went wrong with benchmark_train program')
