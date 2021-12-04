import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mylinearregression import MyLinearRegression as MyLR

def add_polynomial_features(x, power):
    if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] != 1:
        return None
    if not isinstance(power, int) or power <= 0:
        return None
    X = x
    for i in range(1, power):
        X = np.append(X, ((X[:,0] ** (i + 1)).reshape(-1, 1)), axis=1)
    return X

def extract_vectors(filename):
    assert os.path.isfile(filename)
    datas = pd.read_csv(filename)
    x = np.asarray(datas)
    return x

def train_poly_model(x, y, poly_form, theta):
    print('Training model for polynomial form {}'.format(poly_form))
    if not theta:
        theta = [1] * (x.shape[1] + 1)
    my_lr = MyLR(theta, alpha=0.000000001, max_iter=80000)
    my_lr.fit_(x, y)
    y_hat = my_lr.predict_(x)
    mse = my_lr.mse_(y, y_hat)
    return [mse, my_lr]

def get_mse_bar_graph(mse, poly_forms):
    plt.grid()
    plt.bar(poly_forms, mse, width=0.35, zorder=2)
    plt.ylabel('MSE for alpha=1e-9 and max_iter=8e4')
    plt.xlabel('Polynomial form')
    for i, v in enumerate(mse):
        plt.text(poly_forms[i] - 0.17, v + 30, str("{:.0f}".format(v)), fontweight='roman')
    plt.show()

def draw_model_line(my_lr, color, poly_form):
    x_cont = np.linspace(1, 6.5, 1000).reshape(-1, 1)
    x_cont = add_polynomial_features(x_cont, poly_form)
    y_hat = my_lr.predict_(x_cont)
    plt.plot(x_cont[:, 0], y_hat, color=color, label='Line for polynomial model $x^{}$'.format(poly_form), alpha=0.5, zorder=2)

def draw_model_datas(x, y, color):
    plt.scatter(x, y, color=color, label='Dataset', s=20, zorder=2)

def get_models_plot_graph(poly_forms, x_poly, y, my_lr_poly):
    colors_line = ['orange', 'red', 'gold', 'seagreen', 'royalblue', 'mediumorchid']
    plt.grid()
    draw_model_datas(x_poly[0], y, 'dodgerblue')
    for i in poly_forms:
        draw_model_line(my_lr_poly[i - 1], colors_line[i - 1], i)
    plt.ylabel("Standardized score at the spacecraft driving test")
    plt.xlabel("Quantity of blue pills patient has taken (in micrograms)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #Extract datas from .csv file
    datas = extract_vectors('are_blue_pills_magics.csv')
    y = datas[:,2].reshape(-1, 1)
    x_base = datas[:,1].reshape(-1, 1)

    #Get differents polynomial forms of x
    x_poly = []
    poly_forms = []
    for i in range(1, 7):
        x_poly.append(add_polynomial_features(x_base, i))
        poly_forms.append(i)

    #Train model (some thetas are given by subject)
    trained = []
    print('\n--- PERFORMING TRAINING ---\n')
    for i in poly_forms:
        if i == 4:
            theta = [[-20],[ 160],[ -80],[ 10],[ -1]]
        elif i == 5:
            theta = [[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]
        elif i == 6:
            theta = [[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]
        else:
            theta = []
        trained.append(train_poly_model(x_poly[i - 1], y, i, theta))

    #Draw mse bar graph
    print("\n----- MSE BAR GRAPH -----\n")
    mse = []
    for i in poly_forms:
        mse.append(trained[i - 1][0])
        print("MSE for polynomial form {}: {}\n".format(i, mse[i - 1]))
    get_mse_bar_graph(mse, poly_forms)

    #Draw models graphs
    print("\n----- MODELS GRAPH -----\n")
    my_lr_poly= []
    for i in poly_forms:
        my_lr_poly.append(trained[i - 1][1])
    get_models_plot_graph(poly_forms, x_poly, y, my_lr_poly)
