import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mylinearregression import MyLinearRegression as MyLR

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

def extract_vectors(filename):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        x = np.asarray(datas)
        return x
    except:
        print('Something went wrong in extract_vectors function')
        return None

def train_poly_model(x, y, poly_form, theta):
    try:
        print('Training model for polynomial form {}'.format(poly_form))
        if theta.size == 0:
            theta = np.array([1] * (x.shape[1] + 1)).reshape(-1, 1)
        my_lr = MyLR(theta, alpha=0.000000001, max_iter=80000)
        my_lr.fit_(x, y)
        y_hat = my_lr.predict_(x)
        mse = my_lr.mse_(y, y_hat)
        return [mse, my_lr]
    except:
        print('Something went wrong in train_poly_model function')
        return None

def get_mse_bar_graph(mse, poly_forms):
    try:
        plt.grid()
        plt.bar(poly_forms, mse, width=0.35, zorder=2)
        plt.ylabel('MSE for alpha=1e-9 and max_iter=8e4')
        plt.xlabel('Polynomial form')
        for i, v in enumerate(mse):
            plt.text(poly_forms[i] - 0.17, v + 30, str("{:.0f}".format(v)), fontweight='roman')
        plt.show()
    except:
        print('Something went wrong in get_mse_bar_graph function')

def draw_model_line(my_lr, color, poly_form):
    try:
        x_cont = np.linspace(1, 6.5, 1000).reshape(-1, 1)
        x_cont = add_polynomial_features(x_cont, poly_form)
        y_hat = my_lr.predict_(x_cont)
        plt.plot(x_cont[:, 0], y_hat, color=color, label='Line for polynomial model $x^{}$'.format(poly_form), alpha=0.5, zorder=2)
    except:
        print('Something went wrong in draw_model_line function')

def draw_model_datas(x, y, color):
    try:
        plt.scatter(x, y, color=color, label='Dataset', s=20, zorder=2)
    except:
        print('Something went wrong in draw_model_datas function')

def get_models_plot_graph(poly_forms, x_poly, y, my_lr_poly):
    try:
        colors_line = ['orange', 'red', 'gold', 'seagreen', 'royalblue', 'mediumorchid']
        plt.grid()
        draw_model_datas(x_poly[0], y, 'dodgerblue')
        for i in poly_forms:
            draw_model_line(my_lr_poly[i - 1], colors_line[i - 1], i)
        plt.ylabel("Standardized score at the spacecraft driving test")
        plt.xlabel("Quantity of blue pills patient has taken (in micrograms)")
        plt.legend()
        plt.show()
    except:
        print('Something went wrong in get_models_plot_graph function')


if __name__ == '__main__':
    try:
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
                theta = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
            elif i == 5:
                theta = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
            elif i == 6:
                theta = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)
            else:
                theta = np.array([])
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
    except:
        print('Something went wrong in main function')
