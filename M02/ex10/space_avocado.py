from benchmark_train import *
from polynomial_model import add_polynomial_features
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from mylinearregression import MyLinearRegression as MyLR

# Functions to recuperate parameters and infos of best model

def recuperate_x_form(best):
    try:
        form = best['Form'].values[0]
        x1 = int(form[3])
        x2 = int(form[8])
        x3 = int(form[13])
        return (x1, x2, x3)
    except:
        print('Something went wrong with recuperate_x_form function')
        return None

# Functions to recuperate original x set under correct format

def get_original_x_crossed(original_x, form):
    try:
        if form == 'x1*x2 x3':
            original_x = np.c_[np.multiply(original_x[:, 0], original_x[:, 1]), original_x[:, 2]]
        elif form == 'x1*x3 x2':
            original_x = np.c_[np.multiply(original_x[:, 0], original_x[:, 2]), original_x[:, 1]]
        elif form == 'x1 x2*x3':
            original_x = np.c_[original_x[:, 0], np.multiply(original_x[:, 1], original_x[:, 2])]
        else:
            original_x = np.multiply(np.multiply(original_x[:, 0], original_x[:, 1]), original_x[:, 2]).reshape(-1, 1)
        return original_x
    except:
        print('Something went wrong with get_original_x_crossed function')
        return None

def get_original_x_poly(original_x, x1, x2, x3):
    try:
        x1_original = add_polynomial_features(original_x[:, 0].reshape(-1, 1), x1)
        x2_original = add_polynomial_features(original_x[:, 1].reshape(-1, 1), x2)
        x3_original = add_polynomial_features(original_x[:, 2].reshape(-1, 1), x3)
        original_x = np.c_[x1_original[:, :x1], x2_original[:, :x2], x3_original[:, :x3]]
        return original_x
    except:
        print('Something went wrong with get_original_x_poly function')
        return None    

# Function to obtain a line curve in 3D representation

def get_continuous_x(x):
    try:
        x1_cont = np.linspace(x[:, 0].min(), x[:, 0].max(), 10000)
        x2_cont = np.linspace(x[:, 1].min(), x[:, 1].max(), 10000)
        x3_cont = np.linspace(x[:, 2].min(), x[:, 2].max(), 10000)
        return np.c_[x1_cont, x2_cont, x3_cont]
    except:
        print('Something went wrong with get_continuous_x function')
        return None

# Function to display plot for best model

def print_best_representation(model, best, x, y):
    features = ['Weight', 'Production distance', 'Time delivery']

    ## Scatter plot
    # 1 - Get x under correct format for prediction
    x_for_predict = mean_normalization(x)
    if (best['Form'].values[0].find('*') != -1):
        form = best['Form'].values[0]
        x_for_predict = get_original_x_crossed(x_for_predict, form)
    else :
        form = best['Form'].values[0]
        x1, x2, x3 = recuperate_x_form(best)
        x_for_predict = get_original_x_poly(x_for_predict, x1, x2, x3)

    # 2 - Calculate y_hat
    y_hat = model.predict_(x_for_predict)

    # 3 - Scatter plot
    fig = plt.figure(figsize=(10*3, 10))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.scatter(x[:,i], y, color='blue', label='True price', alpha=0.7)
        ax.scatter(x[:,i], y_hat, color='red', label='Predicted price', alpha=0.1)
        ax.set_xlabel(features[i])
        ax.set_ylabel('Price')
        if i == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    plt.show()

    # 3D plots
    # 1 - Create a continuous x for obtaining a continuous line
    x_cont = get_continuous_x(x)

    # 2 - Get x under correct format for prediction
    x_cont_form = mean_normalization(x_cont)
    if (best['Form'].values[0].find('*') != -1):
        form = best['Form'].values[0]
        x_cont_form = get_original_x_crossed(x_cont_form, form)
    else :
        form = best['Form'].values[0]
        x1, x2, x3 = recuperate_x_form(best)
        x_cont_form = get_original_x_poly(x_cont_form, x1, x2, x3)

    # 3 - Calculate y_hat
    y_hat = model.predict_(x_cont_form)

    # 4 - 3D plot
    fig = plt.figure(figsize=(10*3, 10))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        j = i + 1 if i < 2 else 0
        ax.scatter3D(x[:,i].flatten(), x[:, j].flatten(), y.flatten(), color='seagreen', label='True price')
        ax.plot3D(x_cont[:,i].flatten(), x_cont[:, j].flatten(), y_hat.flatten(), color='lime', label='Predicted price')
        ax.set_xlabel(features[i])
        ax.set_ylabel(features[j])
        ax.set_zlabel('Price')
        if i == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=6)
    plt.show()

# Function to train best model

def train_best_model(best, form, x_train, y_train, x_test, y_test):

    # Recuperate form (crossed or polynomial) on x_train and x_test for training
    if (best['Form'].values[0].find('*') != -1):
        form = best['Form'].values[0]
        x_train = get_original_x_crossed(x_train, form)
        x_test = get_original_x_crossed(x_test, form)
    else :
        form = best['Form'].values[0]
        x1, x2, x3 = recuperate_x_form(best)
        x_train = get_original_x_poly(x_train, x1, x2, x3)
        x_test = get_original_x_poly(x_test, x1, x2, x3)

    # Recuperate thetas       
    thetas = (best['Thetas after fit'].values)[0].strip('[]').replace('\' ', '').split(',')
    thetas = [float(i) for i in thetas]

    print('Training best model...\n')

    model = MyLR(thetas, alpha=best['Alpha'].values[0], max_iter=50000)
    model.fit_(x_train, y_train)
    y_hat = model.predict_(x_test)
    print('Loss on testing set: {:.2f}'.format(model.loss_(y_test, y_hat)))
    print('MSE on testing set: {:.2f}'.format(model.mse_(y_test, y_hat)))

    return model

# Functions to draw evaluations curves

def trace_evaluation_curve_mse():
    try:
        df = pd.read_csv('models.csv')
        df = df.replace(to_replace='x\dp1', value='x', regex=True)
        df = df.replace(to_replace='x\dp2', value='x2', regex=True)
        df = df.replace(to_replace='x\dp3', value='x3', regex=True)
        df = df.replace(to_replace='x\dp4', value='x4', regex=True)
        plt.rc('xtick', labelsize=6)
        plt.figure(figsize=(10, 7))
        plt.title('Evaluation curve of MSE for differents tested models')
        plt.plot(['\n'.join(wrap(x, 2)) for x in df['Form']], df['MSE on testing set'])
        plt.xlabel('Model form (powers of: $x_{1} x_{2} x_{3}}$)')
        plt.ylabel('MSE')
        plt.grid()
        plt.show()
    except:
        print('Something went wrong with trace_evaluation_curve function')

def trace_evaluation_curve_loss():
    try:
        df = pd.read_csv('models.csv')
        df = df.replace(to_replace='x\dp1', value='x', regex=True)
        df = df.replace(to_replace='x\dp2', value='x2', regex=True)
        df = df.replace(to_replace='x\dp3', value='x3', regex=True)
        df = df.replace(to_replace='x\dp4', value='x4', regex=True)
        plt.rc('xtick', labelsize=6)
        plt.figure(figsize=(10, 7))
        plt.title('Evaluation curve of loss for differents tested models')
        plt.plot(['\n'.join(wrap(x, 2)) for x in df['Form']], df['Loss on testing set'])
        plt.xlabel('Model form (powers of: $x_{1} x_{2} x_{3}}$)')
        plt.ylabel('MSE')
        plt.grid()
        plt.show()
    except:
        print('Something went wrong with trace_evaluation_curve function')

if __name__ == "__main__":
    # Recuperate best model and testing set
    if os.path.isfile('./models.csv') and os.path.isfile('./sets.npz'):
        print('Models were already trained !')
        best = find_best_model('./models.csv')
        sets = np.load('./sets.npz')
        x, y, x_train, y_train, x_test, y_test = sets['x'], sets['y'], sets['x_train'], sets['y_train'], sets['x_test'], sets['y_test']
    else:
        best, x, y, x_train, y_train, x_test, y_test = launch_benchmark(50000)

    # Draw evaluation curve
    trace_evaluation_curve_loss()
    trace_evaluation_curve_mse()

    # Train and print best model
    model = train_best_model(best, best['Form'].values[0], x_train, y_train, x_test, y_test)
    print_best_representation(model, best, x, y)
