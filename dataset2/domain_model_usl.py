from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import random
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
    # print('loss/regularization--->', loss, regularization)

    return loss #+ regularization


def create_model(df, i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    # params, cov = curve_fit(f, train['concurrent_users'], train['avg_response_time'], bounds=([0,0,0,0,0,0,0],[np.inf,10,np.inf,np.inf,np.inf,np.inf,np.inf]))
    result = minimize(cost, [0,0,1], args=(train['concurrent_users'], train['avg_response_time']), bounds=((0, np.inf), (0, np.inf), (0.00000001, np.inf)))
    # print('result', result)
    print('[RESULT] estimated parameters = ', result.x)
    [s, k, l] = result.x

    response_time_prediction = f(test['concurrent_users'], s, k, l)
    _helpers.print_predictions(test['avg_response_time'].values, response_time_prediction.values)

    rmse = np.sqrt(np.mean(np.square(response_time_prediction - test['avg_response_time'])))
    print('[RESULT] RMSE = ', rmse)

    results_df = pd.DataFrame({'scenario': test['scenario'], 'msg_size': test['msg_size'], 'concurrent_users': test['concurrent_users'], 'avg_response_time': test['avg_response_time'], 'prediction': response_time_prediction})
    print(results_df)

    x = np.arange(0, 1000, 1)
    y = f(x, s, k, l)
    _helpers.plot_curve(df, x, y)    

    return result.x, rmse

def create_models_with_cross_validation():
    error = []
    df = pd.read_csv('summary_truncated.csv', sep=',')
    
    for i in range(5):
        print('\ncase', i+1)
        params, rmse = create_model(df, i)
        error.append(rmse)

    mean_error = _helpers.get_average_error(error)

    return mean_error


def predict(x, params):
    [s, k, l] = params
    y = f(x, s, k, l)

    return y

create_models_with_cross_validation()