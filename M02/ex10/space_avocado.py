import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_datas(filename):
    assert os.path.isfile(filename)
    datas = pd.read_csv(filename)
    x = np.asarray(datas)
    return x[:, 1:x.shape[1] - 1], x[:, x.shape[1] - 1:]

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

if __name__ == '__main__':
    x, y = extract_datas('space_avocado.csv')
    splited_datas = data_spliter(x, y, 0.5)
    print(splited_datas)