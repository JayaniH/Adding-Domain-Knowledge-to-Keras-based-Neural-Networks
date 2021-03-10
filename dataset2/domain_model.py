from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import random
import _helpers

# objective function 1
# def f(X, s, k, l, a1, b1, a2, b2):
#     x1 = X['scenario']
#     x2 = X['msg_size']
#     x3 = X['concurrent_users']

#     y = ((1 + s*(x3-1) + k*x3*(x3-1))/l) * (a1 * x1 + a2 * x2) + (b1 * x1 + b2 * x2)
#     return y

# objective function 2
def f(X, s, k, l, a, b):
    x1 = X['scenario']
    x2 = X['msg_size']
    x3 = X['concurrent_users']

    y = ((1 + s*(x3-1) + k*x3*(x3-1))/l) * (a * x1 * x2) + (b * x1 * x2)
    return y

# objective function 3
# def f(X, s, k, l, a, b):
#     x1 = X['scenario']
#     x2 = X['msg_size']
#     x3 = X['concurrent_users']

#     y = ((1 + s*(x3-1) + k*x3*(x3-1))/l) * (1 + a*x2 + b*x1)
#     return y

def cost(params, X, y_true):
    # print('const func', params, X, y_true)
    [s, k, l, a, b] = params
    y_pred = f(X, s, k, l, a, b)

    #3
    # param_regularization = 10000 * (k * (1 + a + b)/l) + 1000 * (s-k) * (1 + a + b)/l

    #2
    param_regularization = 10000 * ((a * k) / l) + 1000 * (((s - k) * a) / l) + 10 * a 

    #1
    # param_regularization = 100000 * (k * (a1 + a2) / l) + 10 * ((s - k) * (a1 + a2) / l)

    regularization = 2000 * param_regularization
    # print('regularization--->', regularization)
    loss = np.sqrt(np.mean(np.square(y_pred - y_true)))
    # print('pred', y_pred)
    # print('lambda--->', l)
    # print('loss/regularization--->', loss, regularization)
    return loss + regularization


def create_model(df, i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    # params, cov = curve_fit(f, train[['scenario', 'msg_size', 'concurrent_users']], train['avg_response_time'], bounds=([0,0,0,0,0,0,0],[np.inf,10,np.inf,np.inf,np.inf,np.inf,np.inf]))
    result = minimize(cost, [0,0,1,0,0], args=(train[['scenario', 'msg_size', 'concurrent_users']], train['avg_response_time']), bounds=((0, np.inf), (0, np.inf), (0.00000001, np.inf), (0, np.inf), (0, np.inf)))
    # print('result', result)
    print('params--->', result.x)
    [s, k, l, a, b] = result.x

    response_time_prediction = f(test[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a, b)
    _helpers.print_predictions(test['avg_response_time'], response_time_prediction)
    rmse = np.sqrt(np.mean(np.square(response_time_prediction - test['avg_response_time'])))
    print('[RESULT] RMSE = ', rmse)

    for msg in [50, 1024, 10240, 102400]:

        x1 = np.full((1000,), 2)
        x2 = np.full((1000,), msg)
        x3 = np.arange(0, 1000, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        y = f(new_df[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a, b)
        # print('curve_preds--->\n',y)

        df_filtered = df[(df['msg_size'] == msg) & (df['scenario'] == 2)]
        # plt.yscale('log')
        plt.plot(x3, y, label='msg_size='+str(msg))
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Domain Model : scenario = transformation')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    # plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_msg_size.png')
    plt.close()

    for scenario_id in [1,2]:

        scenario = 'passthrough' if scenario_id == 1 else 'transformation'
        x1 = np.full((1000,), scenario_id)
        x2 = np.full((1000,), 1024)
        x3 = np.arange(0, 1000, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        y = f(new_df[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a, b)

        df_filtered = df[(df['scenario'] == scenario_id) & (df['msg_size'] == 1024)]
        # plt.yscale('log')
        plt.plot(x3, y, label='scenario='+str(scenario))
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Domain Model : msg_size = 1024')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    # plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_scenario.png')
    plt.close()

    return result.x, rmse

def create_models_with_cross_validation():
    errors = []
    df = pd.read_csv('summary_truncated1.csv', sep=',')
    
    for i in range(5):
        print('\ncase', i+1)
        params, rmse = create_model(df, i)
        errors.append(rmse)

    mean_error = _helpers.get_average_error(errors)
    return mean_error

def predict(x, params):
    [s, k, l, a, b] = params
    y = f(x, s, k, l, a, b)

    return y


create_models_with_cross_validation()
