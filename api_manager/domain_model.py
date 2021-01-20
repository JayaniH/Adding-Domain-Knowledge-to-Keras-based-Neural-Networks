from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import random

def f(X, s, k, l, a1, b1, a2, b2):
    x1 = X['scenario']
    x2 = X['msg_size']
    x3 = X['concurrent_users']

    return ((1 + s*(x3-1) + k*x3*(x3-1))/l) * (a1 * x1 + a2 * x2) + (b1 * x1 + b2 * x2)

def create_model(df, seed):

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    params, cov = curve_fit(f, train[['scenario', 'msg_size', 'concurrent_users']], train['avg_response_time'], bounds=([0,0,0,0,0,0,0],[np.inf,10,np.inf,np.inf,np.inf,np.inf,np.inf]))
    [s, k, l, a1, b1, a2, b2] = params

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
    plt.title('scenario = passthrough')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    plt.savefig('../../Plots/_api_manager/5_domain_model_test3_randint/' + str(seed+1) + '_msg_size.png')
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
    plt.title('msg_size = 50')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    plt.savefig('../../Plots/_api_manager/5_domain_model_test3_randint/' + str(seed+1) + '_scenario.png')
    plt.close()

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x2, x3, y, c=y)
    # ax.set_xlabel('msg_size')
    # ax.set_ylabel('concurrent_users')
    # ax.set_zlabel('avg_response_time')
    # plt.show()
    # print(df)

    return params, rmse

def get_average_error():
    error = []
    df = pd.read_csv('summary_truncated.csv', sep=',')
    
    for i in range(5):
        print('\ncase', i+1)
        random.seed(i)
        params, rmse = create_model(df, random.randint(0,100))
        error.append(rmse)

    print('\n'.join([str(e) for e in error]), '\n\n')
    print(np.mean(error))

def predict(x, params):
    [s, k, l, a1, b1, a2, b2] = params
    y = f(x, s, k, l, a1, b1, a2, b2)

    return y

# get_average_error()