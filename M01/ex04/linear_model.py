import numpy as np
import os
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR

def extract_vectors(filename):
    assert os.path.isfile(filename)
    datas = np.asarray(pd.read_csv(filename))
    x = datas[:, 1].reshape(-1, 1)
    y = datas[:, 2].reshape(-1, 1)
    return [x, y]

if __name__ == '__main__':
    vectors = extract_vectors("./are_blue_pills_magics.csv")
    x = vectors[0]
    y = vectors[1]
    print(x)
    print(y)
