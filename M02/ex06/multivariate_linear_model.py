import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

#Part One : Univariate linear regression

def extract_vector(filename, vector_name):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        x = np.asarray(datas[vector_name])
        x = x.reshape(-1, 1)
        return x
    except:
        print('Something went wrong in extract_vector function')
        return None

def extract_matrix(filename, features):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        X = np.array(datas[features])
        return X
    except:        
        print('Something went wrong in extract_matrix function')
        return None

def draw_spots(x, y, label, color, size):
    try:
        plt.scatter(x, y, size, label=label, color=color, zorder=2)
    except:
        print('Something went wrong in draw_spots function')

def get_graph(x, y, y_hat, labels, axis, colors):
    try:
        plt.grid()
        draw_spots(x, y, labels[0], colors[0], 45)
        draw_spots(x, y_hat, labels[1], colors[1], 15)
        plt.xlabel(axis[0])
        plt.ylabel(axis[1])
        plt.legend()
        plt.show()
    except:
        print('Something went wrong in get_graph function')

def univar_lr_model(feature, filename, y, theta, labels, alpha, max_iter, axis, colors):
    try:
        x_feat = extract_vector(filename, feature)
        myLR_feat = MyLR(theta, alpha, max_iter)
        myLR_feat.fit_(x_feat, y)
        y_hat_feat = myLR_feat.predict_(x_feat)
        print("Thetas for {}: {}\nMSE for {}: {}\n".format(feature, myLR_feat.theta.tolist(), feature, myLR_feat.mse_(y, y_hat_feat)))
        get_graph(x_feat, y, y_hat_feat, labels, axis, colors)
    except:
        print('Something went wrong in univar_lr_model function')

def univar_lr():
    try:
        #Datas for all
        filename = 'spacecraft_data.csv'
        y = extract_vector(filename, 'Sell_price')
        theta = [1] * 2
        labels = ['Sell price', 'Predicted sell price']

        #Age
        axis_age = ['$x_{1}$: age in years', 'y: sell price (in keuros)']
        colors_age = ['navy', 'dodgerblue']
        univar_lr_model('Age', filename, y, theta, labels, 0.01, 5000, axis_age, colors_age)

        #Thrust
        axis_thrust = ['$x_{2}$: thrust power(in 10KM/s)', 'y: sell price (in keuros)']
        colors_thrust = ['seagreen', 'lime']
        univar_lr_model('Thrust_power', filename, y, theta, labels, 0.0001, 5000, axis_thrust, colors_thrust)

        #Total distance
        axis_distance = ['$x_{3}$: distance totalizer value of spacecraft (in Tmeters))', 'y: sell price (in keuros)']
        colors_distance = ['purple', 'plum']
        univar_lr_model('Terameters', filename, y, theta, labels, 0.0001, 200000, axis_distance, colors_distance)
    except:
        print('Something went wrong in univar_lr function')


# Part Two : Multivariate regression

def draw_sub_spots(ax, x, y, label, color, size):
    try:
        ax.scatter(x, y, size, label=label, color=color, zorder=2)
    except:
        print('Something went wrong in draw_sub_spots function')

def get_sub_graph(ax, fig, x, y, y_hat, labels, axis, colors):
    try:
        ax.grid()
        draw_sub_spots(ax, x, y, labels[0], colors[0], 45)
        draw_sub_spots(ax, x, y_hat, labels[1], colors[1], 15)
        ax.set_xlabel(axis[0])
        ax.set_ylabel(axis[1])
        ax.legend()
    except:
        print('Something went wrong in get_sub_graph function')

def multivar_lr_model(filename, y, theta, labels):
    try:
        X = extract_matrix(filename, ['Age','Thrust_power','Terameters'])
        myLR = MyLR(theta, alpha=0.00007, max_iter=900000)
        myLR.fit_(X, y)
        y_hat = myLR.predict_(X)
        print("Thetas for multivariate model: {}".format(myLR.theta.tolist()))
        print("MSE for multivariate model: {}\n".format(myLR.mse_(y, y_hat)))
        
        fig, ax = plt.subplots(1, 3)

        #Graph for age
        axis_age = ['$x_{1}$: age in years', 'y: sell price (in keuros)']
        colors_age = ['navy', 'dodgerblue']
        get_sub_graph(ax[0], fig, X[:,0], y, y_hat, labels, axis_age, colors_age)

        #Graph for thrust
        axis_thrust = ['$x_{2}$: thrust power(in 10KM/s)', 'y: sell price (in keuros)']
        colors_thrust = ['seagreen', 'lime']
        get_sub_graph(ax[1], fig, X[:,1], y, y_hat, labels, axis_thrust, colors_thrust)

        #Graph for total distance
        axis_distance = ['$x_{3}$: distance totalizer value of spacecraft (in Tmeters))', 'y: sell price (in keuros)']
        colors_distance = ['purple', 'plum']
        get_sub_graph(ax[2], fig, X[:,2], y, y_hat, labels, axis_distance, colors_distance)

        plt.show()
    except:
        print("Something went wrong in multivar_lr_model function")

def multivar_lr():
    try:
        filename = 'spacecraft_data.csv'
        y = extract_vector(filename, 'Sell_price')
        theta = [1] * 4
        labels = ['Sell price', 'Predicted sell price']
        multivar_lr_model(filename, y, theta, labels)
    except:
        print('Something went wrong in multivar_lr function')


import sklearn.metrics
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':

    print("WARNING: Calculations and graph loading may take some time\n")

    print("----------------------------")
    print("UNIVARIATE LINEAR REGRESSION")
    print("----------------------------")
    univar_lr()

    print("----------------------------")
    print("MULTIVARIATE LINEAR REGRESSION")
    print("----------------------------")
    multivar_lr()

    print("----------------------------")
    print("SUBJECT TESTS")
    print("----------------------------")
    print('Univariate (age example):')
    data = pd.read_csv("spacecraft_data.csv")
    X = np.array(data[['Age']])
    Y = np.array(data[['Sell_price']])
    myLR_age = MyLR(theta = [[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
    myLR_age.fit_(X[:,0].reshape(-1,1), Y)
    y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
    print('MSE for age: ', myLR_age.mse_(y_pred,Y))
    #Output
    #55736.86719...

    print('\nMultivariate:')
    print('Can\'t perform example of subject for multivariate because alpha is too big for my fit_ algorithm...')

