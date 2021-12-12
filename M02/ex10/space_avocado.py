import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from mylinearregression import MyLinearRegression as MyLR

def extract_datas(filename):
    assert os.path.isfile(filename)
    datas = pd.read_csv(filename)
    x = np.asarray(datas)
    return x[:, 1:x.shape[1] - 1].reshape(-1, 3), x[:, x.shape[1] - 1:].reshape(-1, 1)

def find_best_model(filename):
    assert os.path.isfile(filename)
    df = pd.read_csv(filename)
    min_col = df.min()
    best = df.loc[df['Global MSE'] == min_col['Global MSE']]
    print('\033[92mBest model:\033[0m Form \033[34m{}\033[0m of MSE \033[34m{}\033[0m'.format(best['Form'].values[0], best['Global MSE'].values[0]))
    return best

def recuperate_thetas(best):
    thetas = (best['Thetas after fit'].values)[0].strip('[]').replace('\' ', '').split(',')
    thetas = [float(i) for i in thetas]
    return thetas

def recuperate_lr(best, thetas):
    alpha = ((best['Alpha']).values)[0]
    return MyLR(thetas, alpha=alpha)

def recuperate_x_form(best):
    form = best['Form'].values[0]
    x1 = int(form[3])
    x2 = int(form[8])
    x3 = int(form[13])
    return (x1, x2, x3)

def add_polynomial_features(x, power):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    if not isinstance(power, int) or power <= 0:
        return None
    X = x
    for i in range(1, power):
        X = np.append(X, ((X[:,0] ** (i + 1)).reshape(-1, 1)), axis=1)
    return X

def get_continuous_x(x):
    x1_cont = np.linspace(x[:, 0].min(), x[:, 0].max(), 10000)
    x2_cont = np.linspace(x[:, 1].min(), x[:, 1].max(), 10000)
    x3_cont = np.linspace(x[:, 2].min(), x[:, 2].max(), 10000)
    return np.c_[x1_cont, x2_cont, x3_cont]

def get_poly_continuous_x(x, x1, x2, x3):
    x1_cont = np.linspace(x[:, 0].min(), x[:, 0].max(), 10000)
    x2_cont = np.linspace(x[:, 1].min(), x[:, 1].max(), 10000)
    x3_cont = np.linspace(x[:, 2].min(), x[:, 2].max(), 10000)
    if x1 != 1:
        x1_cont = add_polynomial_features(x1_cont.reshape(-1, 1), x1)
    if (x2 != 1):
        x2_cont = add_polynomial_features(x2_cont.reshape(-1, 1), x2)
    if (x3 != 1):
        x3_cont = add_polynomial_features(x3_cont.reshape(-1, 1), x3)
    return np.c_[x1_cont, x2_cont, x3_cont]

def get_cross_continuous_x(x, form):
    if form == 'x1*x2 x3':
        return np.c_[np.multiply(x[:, 0], x[:, 1]), x[:, 2]]
    elif form == 'x1*x3 x2':
        return np.c_[np.multiply(x[:, 0], x[:, 2]), x[:, 1]]
    elif form == 'x1 x2*x3':
        return np.c_[x[:, 0], np.multiply(x[:, 1], x[:, 2])]
    return np.multiply(np.multiply(x[:, 0], x[:, 1]), x[:, 2]).reshape(-1, 1)

def print_best_representation(best, x, y):
    thetas = recuperate_thetas(best)
    best_lr = recuperate_lr(best, thetas)
    if (best['Form'].values[0].find('*') != -1):
        form = best['Form'].values[0]
    else :
        x1, x2, x3 = recuperate_x_form(best)
    x_cont = get_continuous_x(x)
    if (best['Form'].values[0].find('*') != -1):
        x_cont_poly = get_cross_continuous_x(x_cont, form)
    else :
        x_cont_poly = get_poly_continuous_x(x_cont, x1, x2, x3)
    y_hat = best_lr.predict_(x_cont_poly)
    print(y_hat.shape)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:,0].flatten(), x[:, 2].flatten(), y.flatten(), color='seagreen', label='True price')
    #ax.scatter3D(x_cont[:,0].flatten(), x_cont[:, 2].flatten(), y_hat.flatten(), color='lime', label='Predicted price')
    ax.plot3D(x_cont[:,0].flatten(), x_cont[:, 2].flatten(), y_hat.flatten(), color='lime', label='Predicted price')
    ax.set_xlabel('Weight')
    ax.set_ylabel('Production distance')
    ax.set_zlabel('Price')
    ax.legend()
    plt.show()

def trace_evaluation_curve(filename):
    df = pd.read_csv(filename)
    df = df.replace(to_replace='x\dp1', value='x', regex=True)
    df = df.replace(to_replace='x\dp2', value='x2', regex=True)
    df = df.replace(to_replace='x\dp3', value='x3', regex=True)
    df = df.replace(to_replace='x\dp4', value='x4', regex=True)
    plt.rc('xtick', labelsize=6)
    plt.figure(figsize=(10, 7))
    plt.title('Evaluation curve of MSE for differents tested models')
    plt.plot(['\n'.join(wrap(x, 2)) for x in df['Form']], df['Global MSE'])
    plt.xlabel('Model form (powers of: $x_{1} x_{2} x_{3}}$)')
    plt.ylabel('MSE')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    x, y = extract_datas('space_avocado.csv')
    trace_evaluation_curve('models.csv')
    best = find_best_model('models.csv')
    print_best_representation(best,x, y)