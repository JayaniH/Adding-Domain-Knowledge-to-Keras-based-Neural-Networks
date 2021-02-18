from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import random

# USL function
def f(n, s, k, l):
	return (1 + s*(n-1) + k*n*(n-1))/l


def cost(params, X, y_true):
    # print('const func', params, X, y_true)
    [s, k, l] = params
    y_pred = f(X, s, k, l)

    param_regularization = 100000 * k/l + 100 *(s-k)/l
    regularization = 1 * param_regularization
    # print('regularization--->', regularization)

    loss = np.sqrt(np.mean(np.square(y_pred - y_true)))
    # print('loss/regularization--->', loss, regularization)

    return loss + regularization


def create_model(df, i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    # params, cov = curve_fit(f, train['concurrent_users'], train['avg_response_time'], bounds=([0,0,0,0,0,0,0],[np.inf,10,np.inf,np.inf,np.inf,np.inf,np.inf]))
    result = minimize(cost, [0,0,1], args=(train['concurrent_users'], train['avg_response_time']), bounds=((0, np.inf), (0, np.inf), (0.00000001, np.inf)))
    # print('result', result)
    print('params--->', result.x)
    [s, k, l] = result.x

    predY = f(test['concurrent_users'], s, k, l)
    rmse = np.sqrt(np.mean(np.square(predY - test['avg_response_time'])))
    print('\navg_response_time:\n','\n'.join([str(val) for val in test['avg_response_time'].values]))
    print('\npredicted avg_response_time:\n', '\n'.join([str(val) for val in predY.values]))
    print('RMSE -> ', rmse)

    results_df = pd.DataFrame({'scenario': test['scenario'], 'msg_size': test['msg_size'], 'concurrent_users': test['concurrent_users'], 'avg_response_time': test['avg_response_time'], 'prediction': predY})
    print(results_df)

    x = np.arange(0, 1000, 1)
    y = f(x, s, k, l)
    # print(y)
    plt.scatter(df['concurrent_users'], df['avg_response_time'], label='actual data')
    plt.plot(x, y, label='forecast')

    plt.title('Domain Model')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    # plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_msg_size.png')
    plt.close()

    return result.x, rmse

def get_average_error():
    error = []
    df = pd.read_csv('summary_truncated.csv', sep=',')
    
    for i in range(5):
        print('\ncase', i+1)
        params, rmse = create_model(df, i)
        error.append(rmse)

    print('\n'.join([str(e) for e in error]), '\n\n')
    print('mean error --->', np.mean(error))

def predict(x, params):
    [s, k, l] = params
    y = f(x, s, k, l)

    return y

def test_regularization():
    df = pd.read_csv('summary_truncated.csv', sep=',')
    
    for c in [1000]:

        df_filtered = df[df['concurrent_users'] == c]

        X = df_filtered['concurrent_users']
        Y = df_filtered['avg_response_time']

        print('concurrent_users = ', c, '\n')
        # print(X['concurrent_users'])

        result = minimize(cost, [0,0,1,0,0,0,0], args=(X, Y, 1, 1), bounds=((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)))
        # print('result', result)
        print('params--->', result.x)
        [s, k, l] = result.x

# get_average_error()

# test_regularization()