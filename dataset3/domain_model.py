from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.optimize import minimize

# USL function
def f(n, s, k, l):
	return (1 + s*(n-1) + k*n*(n-1))/l


def cost(params, X, y_true):
    [s, k, l] = params
    y_pred = f(X, s, k, l)

    param_regularization = 100000 * k/l + 100 *(s-k)/l
    regularization = 1 * param_regularization
    # print('regularization--->', regularization)

    loss = np.sqrt(np.mean(np.square(y_pred - y_true)))
    # print('loss/regularization--->', loss, regularization)

    return loss #+ regularization


def create_model(df, train, test, i):

    params, cov = curve_fit(f, train['concurrent_users'], train['mid_latency'], bounds=([0,0,0],[np.inf,10,np.inf]))
    # result = minimize(cost, [0,0,1], args=(train['concurrent_users'], train['mid_latency']), bounds=((0, np.inf), (0, np.inf), (0.00000001, np.inf)))
    # print('result', result)
    print('params--->', params)
    [s, k, l] = params
    predY = f(test['concurrent_users'], s, k, l)

    rmse = np.sqrt(np.mean(np.square(predY - test['mid_latency'])))
    mae = np.mean(np.abs(predY - test['mid_latency']))
    mape = np.mean(np.abs((predY - test['mid_latency'])/test['mid_latency']))*100 

    print('\nlatency:\n','\n'.join([str(val) for val in test['mid_latency'].values]))
    print('\npredicted latency by USL domain model:\n', '\n'.join([str(val) for val in predY.values]))
    
    print('[INFO] rmse/mae/mape...', rmse, mae, mape, '\n')

    # results_df = pd.DataFrame({'concurrent_users': test['concurrent_users'], 'cores': test['cores'], 'workload_mix': test['workload_mix'], 'mid_latency': test['mid_latency'], 'prediction': predY})
    # print(results_df)

    x = np.arange(0, 200, 1)
    y = f(x, s, k, l)
    # print(y)
    plt.scatter(train['concurrent_users'], train['mid_latency'], label='actual data')
    plt.plot(x, y, label='forecast')

    plt.title('[curve_fit]\nDomain Model')
    plt.xlabel('concurrent_users')
    plt.ylabel('mid_latency')
    plt.legend()
    # plt.show()
    # plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_cores.png')
    plt.close()

    return params, rmse, mae, mape

def get_average_error():
    error_rmse = []
    error_mae = []
    error_mape = []

    estimate_s = []
    estimate_k = []
    estimate_l = []

    df = pd.read_csv('tpcw_concurrency.csv', sep=',')
    
    print('[INFO] constructing k fold split...')
    kf = KFold(n_splits=5, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('\nK', i+1)
        train = df.iloc[train_i]
        test = df.iloc[test_i]
        params, rmse, mae, mape = create_model(df, train, test, i)

        [s, k, l] = params
        estimate_s.append(s)
        estimate_k.append(k)
        estimate_l.append(l)

        error_rmse.append(rmse)
        error_mae.append(mae)
        error_mape.append(mape)
        i += 1

    print("\nParameter Estimates - Sigma(s)\n")
    print("\n".join([str(s) for s in estimate_s]), "\n\n")
    print("Parameter Estimates - Kappa(k)\n")
    print("\n".join([str(k) for k in estimate_k]), "\n\n")
    print("Parameter Estimates - Lambda(l)\n")
    print("\n".join([str(l) for l in estimate_l]), "\n\n")
    
    print('Prediction Errors\n')
    print('\n'.join([str(e) for e in error_rmse]), '\n')
    print('mean rmse --->', np.mean(error_rmse), '\n\n')
    print('\n'.join([str(e) for e in error_mae]), '\n')
    print('mean mae --->', np.mean(error_mae), '\n\n')
    print('\n'.join([str(e) for e in error_mape]), '\n')
    print('mean mape --->', np.mean(error_mape), '\n\n')

def get_average_error1():
    df = pd.read_csv('tpcw_concurrency.csv', sep=',')

    print('[INFO] constructing train/test split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=14)

    params, rmse, mae, mape = create_model(df, train, test, 0)


def predict(x, params):
    [s, k, l] = params
    y = f(x, s, k, l)

    return y

get_average_error()