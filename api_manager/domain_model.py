from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import random

# def f(X, s, k, l, a1, b1, a2, b2):
#     x1 = X['scenario']
#     x2 = X['msg_size']
#     x3 = X['concurrent_users']

#     y = ((1 + s*(x3-1) + k*x3*(x3-1))/l) * (a1 * x1 + a2 * x2) + (b1 * x1 + b2 * x2)
#     return y

def f(X, s, k, l, a1, b1, a2, b2):
    x1 = X['scenario']
    x2 = X['msg_size']
    x3 = X['concurrent_users']

    y = (b1 * x1 + b2 * x2 + s/l) * x3
    return y

def cost(params, X, y_true, upper, lower):
    # print('const func', params, X, y_true)
    [s, k, l, a1, b1, a2, b2] = params
    y_pred = f(X, s, k, l, a1, b1, a2, b2)
    limit_regularization = np.mean(np.maximum(0.0, (y_pred - upper))) + np.mean(np.maximum(0.0, (lower - y_pred)))
    param_regularization = 5 * np.square(a1) + 5 * np.square(a2) + 5 * np.square(k) + 5 * np.square(b1) + 5 * np.square(b2) + 5 * np.square(s)
    regularization = param_regularization
    # print('regularization--->', regularization)
    loss = np.sqrt(np.mean(np.square(y_pred - y_true))) + regularization
    # print('pred', y_pred)
    return loss


def create_model(df, i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    upper = np.mean(df['avg_response_time']) + 0.5 * np.std(df['avg_response_time'])
    lower = np.mean(df['avg_response_time']) - 0.01 * np.std(df['avg_response_time'])
    print(np.mean(df['avg_response_time']))
    print(np.std(df['avg_response_time']))
    print('upper/lower=', upper, lower)

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    # params, cov = curve_fit(f, train[['scenario', 'msg_size', 'concurrent_users']], train['avg_response_time'], bounds=([0,0,0,0,0,0,0],[np.inf,10,np.inf,np.inf,np.inf,np.inf,np.inf]))
    result = minimize(cost, [1,0,1,0,0,0,0], args=(train[['scenario', 'msg_size', 'concurrent_users']], train['avg_response_time'], upper, lower), bounds=((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)))
    # print('result', result)
    print('params--->', result.x)
    [s, k, l, a1, b1, a2, b2] = result.x

    predY = f(test[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a1, b1, a2, b2)
    rmse = np.sqrt(np.mean(np.square(predY - test['avg_response_time'])))
    print('\navg_response_time:\n','\n'.join([str(val) for val in test['avg_response_time'].values]))
    print('\npredicted avg_response_time:\n', '\n'.join([str(val) for val in predY.values]))
    print('RMSE -> ', rmse)

    x_max = max(max(df['msg_size']), max(df['concurrent_users']))
    size = max(df['concurrent_users'])

    for msg in [50, 1024, 10240, 102400]:

        x1 = np.full((1000,), 1)
        x2 = np.full((1000,), msg)
        x3 = np.arange(0, 1000, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        y = f(new_df[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a1, b1, a2, b2)

        df_filtered = df[(df['msg_size'] == msg) & (df['scenario'] == 1)]
        # plt.yscale('log')
        plt.plot(x3, y, label='msg_size='+str(msg))
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Domain Model : scenario = passthrough')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    plt.show()
    # plt.savefig('../../Plots/_api_manager/13_domain_model_minimization_regularization_upper_mean_5std_lower_mean_01std/' + str(i+1) + '_msg_size.png')
    plt.close()

    for scenario_id in [1,2]:

        scenario = 'passthrough' if scenario_id == 1 else 'transformation'
        x1 = np.full((1000,), scenario_id)
        x2 = np.full((1000,), 50)
        x3 = np.arange(0, 1000, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        y = f(new_df[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a1, b1, a2, b2)

        df_filtered = df[(df['scenario'] == scenario_id) & (df['msg_size'] == 50)]
        # plt.yscale('log')
        plt.plot(x3, y, label='scenario='+str(scenario))
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Domain Model : msg_size = 50')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    plt.show()
    # plt.savefig('../../Plots/_api_manager/13_domain_model_minimization_regularization_upper_mean_5std_lower_mean_01std/' + str(i+1) + '_scenario.png')
    plt.close()

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x2, x3, y, c=y)
    # ax.set_xlabel('msg_size')
    # ax.set_ylabel('concurrent_users')
    # ax.set_zlabel('avg_response_time')
    plt.show()
    # print(df)

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
    [s, k, l, a1, b1, a2, b2] = params
    y = f(x, s, k, l, a1, b1, a2, b2)

    return y

get_average_error()