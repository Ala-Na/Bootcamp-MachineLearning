import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from my_linear_regression import MyLinearRegression as MyLR

def extract_vectors(filename):
    assert os.path.isfile(filename)
    try:
        datas = np.asarray(pd.read_csv(filename))
        x = datas[:, 1].reshape(-1, 1)
        y = datas[:, 2].reshape(-1, 1)
        return [x, y]
    except:
        return None

def draw_hypothesis(x, y, y_hat):
    try:
        plt.grid(visible='yes')
        plt.scatter(x, y, c='deepskyblue', label='$S_{true}(pills)$')
        plt.plot(x, y_hat, c='limegreen', linestyle='--', marker = 'X', label ='$S_{predict}(pills)$')
        plt.xlabel('Quantity of blue pill (in micrograms)')
        plt.ylabel('Space driving score')
        leg = plt.legend(frameon=False, bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', ncol=4, borderaxespad=0)
        plt.show()
    except:
        print("Something went wrong")

def evolution_loss_function(x, y):
    try:
        plt.grid(visible='yes')
        i = 0
        for theta_0 in np.linspace(80, 100, 6):
            J_theta0 = []
            thetas_1 = []
            for theta_1 in np.linspace(theta_0 - 500, theta_0 + 100, 3000):
                linear_model = MyLR(np.array([[theta_0], [theta_1]]))
                J_theta0.append(linear_model.loss_(y, linear_model.predict_(x)))
                thetas_1.append(theta_1)
            plt.plot(thetas_1, J_theta0, color="{}".format(i/10 + 0.3), label='$J(({}_{}=c_{}{}_{})$'.format('\\theta', 0, i, '\\theta', 1))
            i += 1
        plt.xlim([-15, -3])
        plt.ylim([10, 150])
        leg = plt.legend(loc='lower right')
        plt.show()
    except:
        print("Something went wrong")
