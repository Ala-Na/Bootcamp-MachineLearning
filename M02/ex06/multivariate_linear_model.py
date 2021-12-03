import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR

#Part One : Univariate regression

def extract_vector(filename, vector_name):
    assert os.path.isfile(filename)
    datas = pd.read_csv(filename)
    x = np.asarray(datas[vector_name])
    x = x.reshape(-1, 1)
    return x

def draw_spots(x, y, label, color, size):
    plt.scatter(x, y, size, label=label, color=color, zorder=2)

def get_graph(x, y, y_hat, labels, axis, colors):
    plt.grid()
    draw_spots(x, y, labels[0], colors[0], 45)
    draw_spots(x, y_hat, labels[1], colors[1], 15)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    plt.legend()
    plt.show()

def univar_lr_model(feature, filename, y, thetas, labels, alpha, max_iter, axis, colors):
    x_feat = extract_vector(filename, feature)
    myLR_feat = MyLR(thetas, alpha, max_iter)
    myLR_feat.fit_(x_feat, y)
    y_hat_feat = myLR_feat.predict_(x_feat)
    print("Thetas for {}: {}\nMSE for {}: {}\n".format(feature, myLR_feat.thetas.tolist(), feature, myLR_feat.mse_(y, y_hat_feat)))
    get_graph(x_feat, y, y_hat_feat, labels, axis, colors)

def univar_lr():
    #Datas for all
    filename = 'spacecraft_data.csv'
    y = extract_vector(filename, 'Sell_price')
    thetas = [1] * (y.shape[1] + 1)
    labels = ['Sell price', 'Predicted sell price']

    #Age
    axis_age = ['$x_{1}$: age in years', 'y: sell price (in keuros)']
    colors_age = ['navy', 'dodgerblue']
    univar_lr_model('Age', filename, y, thetas, labels, 0.01, 5000, axis_age, colors_age)

    #Thrust
    axis_thrust = ['$x_{2}$: thrust power(in 10KM/s)', 'y: sell price (in keuros)']
    colors_thrust = ['seagreen', 'lime']
    univar_lr_model('Thrust_power', filename, y, thetas, labels, 0.0001, 5000, axis_thrust, colors_thrust)

    #Total distance
    axis_distance = ['$x_{3}$: distance totalizer value of spacecraft (in Tmeters))', 'y: sell price (in keuros)']
    colors_distance = ['purple', 'plum']
    univar_lr_model('Terameters', filename, y, thetas, labels, 0.0001, 200000, axis_distance, colors_distance)

# Part Two : Multivariate regression

def multivar_lr():
    

if __name__ == '__main__':
    print("----------------------------")
    print("UNIVARIATE LINEAR REGRESSION")
    print("----------------------------")
    univar_lr()
    print("MULTIVARIATE LINEAR REGRESSION")
    print("----------------------------")