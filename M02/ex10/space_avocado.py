import numpy as np
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
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
    return best

def get_y_hat(best):
    thetas = (best['Thetas after fit'].values)[0].strip('[]').replace('\' ', '').split(',')
    thetas = [float(i) for i in thetas]
    alpha = ((best['Alpha']).values)[0]
    max_iter = (best['Max iter'].values)[0]
    best_lr = MyLR(thetas, alpha=alpha, max_iter=((int)(max_iter)))


if __name__ == "__main__":
    x, y = extract_datas('space_avocado.csv')
    best = find_best_model('models.csv')
    print(best)
    get_y_hat(best)