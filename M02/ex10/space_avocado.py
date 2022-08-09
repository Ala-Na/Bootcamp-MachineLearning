import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from textwrap import wrap
from mylinearregression import MyLinearRegression as MyLR

def extract_datas(filename):
    assert os.path.isfile(filename)
    try:
        datas = pd.read_csv(filename)
        x = np.asarray(datas)
        return x[:, 1:x.shape[1] - 1].reshape(-1, 3), x[:, x.shape[1] - 1:].reshape(-1, 1)
    except:
        print('Something went wrong with extract_datas function')
        return None

def find_best_model(filename):
    assert os.path.isfile(filename)
    try:
        df = pd.read_csv(filename)
        min_col = df.min()
        best = df.loc[df['Global MSE'] == min_col['Global MSE']]
        print('\033[92mBest model:\033[0m Form \033[34m{}\033[0m of MSE \033[34m{}\033[0m'.format(best['Form'].values[0], best['Global MSE'].values[0]))
        print('Thetas \033[34m{}\033[0m'.format(best['Thetas after fit'].values[0]))
        return best
    except:
        print('Something went wrong with find_best_model function')
        return None

def recuperate_thetas(best):
    try:
        thetas = (best['Thetas after fit'].values)[0].strip('[]').replace('\' ', '').split(',')
        thetas = [float(i) for i in thetas]
        return thetas
    except:
        print('Something went wrong with recuperate_thetas function')
        return None

def recuperate_lr(best, thetas):
    try:
        alpha = ((best['Alpha']).values)[0]
        return MyLR(thetas, alpha=alpha, max_iter=1000000)
    except:
        print('something went wrong with recuperate_lr function')
        return None

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


def get_continuous_x(x):
    try:
        x1_cont = np.linspace(x[:, 0].min(), x[:, 0].max(), 10000)
        x2_cont = np.linspace(x[:, 1].min(), x[:, 1].max(), 10000)
        x3_cont = np.linspace(x[:, 2].min(), x[:, 2].max(), 10000)
        return np.c_[x1_cont, x2_cont, x3_cont]
    except:
        print('Something went wrong with get_continuous_x function')
        return None

def get_poly_continuous_x(x, x1, x2, x3):
    try:
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
    except:
        print('Something went wrong with get_poly_continuous_x function')
        return None

def get_cross_continuous_x(x, form):
    try:
        if form == 'x1*x2 x3':
            return np.c_[np.multiply(x[:, 0], x[:, 1]), x[:, 2]]
        elif form == 'x1*x3 x2':
            return np.c_[np.multiply(x[:, 0], x[:, 2]), x[:, 1]]
        elif form == 'x1 x2*x3':
            return np.c_[x[:, 0], np.multiply(x[:, 1], x[:, 2])]
        return np.multiply(np.multiply(x[:, 0], x[:, 1]), x[:, 2]).reshape(-1, 1)
    except:
        print('Something went wrong with get_cross_continuous_x function')
        return None

# Save training to models file

def add_model_to_file(name, global_mse, thetas, alpha):
    try:
        df = pd.DataFrame()
        df = df.append({'Form' : name,'Global MSE' : global_mse, \
            'Thetas after fit' : thetas[:, 0].tolist(), 'Alpha' : alpha}, ignore_index=True)
        df.to_csv('models.csv', mode='a', header=False)
    except:
        print('Something went wrong with add_model_to_file function') 

def save_training(form, mse, thetas, alpha):
    try:
        df = pd.read_csv('models.csv')
        find_form = df.loc[df['Form'] == form]
        if find_form.empty:
            add_model_to_file(form, mse, thetas, alpha)
            return
        if mse == float("inf") or find_form['Global MSE'].values[0] <= mse:
            return
        idx = (find_form.index.tolist())[0]
        df.drop(idx, inplace=True)
        df.to_csv('models.csv', index=False)
        add_model_to_file(form, mse, thetas, alpha)
    except:
        print("Something went wrong with save_training function")

def print_best_representation(best, x, y):
    try:
        thetas = recuperate_thetas(best)
        best_lr = recuperate_lr(best, thetas)
        if (best['Form'].values[0].find('*') != -1):
            form = best['Form'].values[0]
            original_x_set = get_original_x_crossed(x, form)
        else :
            form = best['Form'].values[0]
            x1, x2, x3 = recuperate_x_form(best)
            original_x_set = get_original_x_poly(x, x1, x2, x3)

        print('\nTraining best model... (can take some time)\n')
        best_lr.fit_(original_x_set, y)
        y_hat_fit = best_lr.predict_(original_x_set)
        mse_fit = best_lr.mse_(y_hat_fit, y)
        print('\033[92mAfter training\033[0m MSE \033[34m{}\033[0m'.format(mse_fit))
        print('Thetas \033[34m{}\033[0m'.format(best_lr.theta))
        save_training(form, mse_fit, best_lr.theta, best_lr.alpha)

        x_cont = get_continuous_x(x)
        if (best['Form'].values[0].find('*') != -1):
            x_cont_poly = get_cross_continuous_x(x_cont, form)
        else :
            x_cont_poly = get_poly_continuous_x(x_cont, x1, x2, x3)

        y_hat = best_lr.predict_(x_cont_poly)
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
    except:
        print('Something went wrong with print_best_representation function')

def trace_evaluation_curve(filename):
    try:
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
    except:
        print('Something went wrong with trace_evaluation_curve function')

if __name__ == "__main__":
    try:
        x, y = extract_datas('space_avocado.csv')
        trace_evaluation_curve('models.csv')
        best = find_best_model('models.csv')
        print_best_representation(best,x, y)
    except:
        print('Something went wrong with space_avocado program. Please check that \
            a models.csv file is present or launch banchmark_train.py')
