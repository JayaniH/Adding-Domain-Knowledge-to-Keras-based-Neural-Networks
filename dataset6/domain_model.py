from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import dataset
import _helpers

# USL function
def f(n, s, k, l):
	return (1 + s*(n-1) + k*n*(n-1))/l


def cost(params, X, y_true):
    [s, k, l] = params
    y_pred = f(X, s, k, l)

    param_regularization = 100000 * k/l + 100 *(s-k)/l
    regularization = 1 * param_regularization

    loss = np.sqrt(np.mean(np.square(y_pred - y_true)))
    # print('[INFO] Loss / Regularization--->', loss, regularization)

    return loss # + regularization


def create_model(df, train_i, test_i, i):

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    params, _ = curve_fit(f, train['concurrent_users'], train['latency'], bounds=([0,0,1e-9],[1,1,np.inf]))
    # result = minimize(cost, [0,0,1], args=(train['concurrent_users'], train['latency']), bounds=((1e-9, np.inf), (1e-9, np.inf), (1e-9, np.inf)))
    # print('result', result)

    print('[RESULT] Estimated Parameters = ', params)
    [s, k, l] = params
    predY = f(test['concurrent_users'], s, k, l) 

    _helpers.print_predictions(test['latency'].values, predY.values)

    rmse, mae, mape = _helpers.get_error(test['latency'], predY)

    x = np.arange(0, 1000, 1)
    y = f(x, s, k, l)
    # _helpers.plot_curve(train, x, y)    

    return params, rmse, mae, mape


def create_model_with_cross_validation():
    errors = {'rmse' : [], 'mae': [], 'mape': []}
    param_estimates = {'s': [], 'k': [], 'l': []}

    df = pd.read_csv('Ballerina_Dataset_truncated.csv', sep=',')
    df = dataset.remove_outliers1(df)
    
    print('[INFO] Constructing k fold split...')
    kf = KFold(n_splits=10, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('-----------------------------------------------------')
        print('\nK', i+1)
        params, rmse, mae, mape = create_model(df, train_i, test_i, i)

        [s, k, l] = params
        param_estimates['s'].append(s)
        param_estimates['k'].append(k)
        param_estimates['l'].append(l)

        errors['rmse'].append(rmse)
        errors['mae'].append(mae)
        errors['mape'].append(mape)
        i += 1

    _helpers.print_parameters_and_errors(param_estimates, errors)

    return param_estimates, errors


def get_parameters():
    param_estimates = []

    df = pd.read_csv('Ballerina_Dataset_truncated.csv', sep=',')
    df = dataset.remove_outliers1(df)
    kf = KFold(n_splits=10, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('-----------------------------------------------------')
        print('\nK', i+1)
        params, _, _, _ = create_model(df, train_i, test_i, i)

        param_estimates.append(params)
        
        i += 1

    return param_estimates


def evaluate_model():
    df = pd.read_csv('Ballerina_Dataset_truncated.csv', sep=',')
    df = dataset.remove_outliers1(df)

    params = get_parameters()
    [s, k, l] = params
    print('[RESULT] Estimated patameters = ', params)

    predY = f(df['concurrent_users'], s, k, l)

    _helpers.print_predictions(df['latency'].values, predY.values)

    rmse, mae, mape = _helpers.get_error(df['latency'], predY)

    x = np.arange(0, 200, 1)
    y = f(x, s, k, l)
    _helpers.plot_curve(df, x, y)

    return rmse, mae, mape


def predict(x, params):
    [s, k, l] = params
    y = f(x, s, k, l)

    return y


create_model_with_cross_validation()
# print(get_parameters())
# evaluate_model()