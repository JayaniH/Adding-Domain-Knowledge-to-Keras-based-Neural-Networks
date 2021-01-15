from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit

def f(X, s, k, l, a1, b1, a2, b2):
    x1 = X['scenario']
    x2 = X['msg_size']
    x3 = X['concurrent_users']

    return ((1 + s*(x3-1) + k*x3*(x3-1))/l) * (a1 * x1 + a2 * x2) + (b1 * x1 + b2 * x2)

def create_model():
    df = pd.read_csv('summary_truncated.csv', sep=',')

    (train, test) = train_test_split(df, test_size=0.3)

    params, cov = curve_fit(f, train[['scenario', 'msg_size', 'concurrent_users']], train['avg_response_time'], bounds=([0,0,0,0,0,0,0],[np.inf,10,np.inf,np.inf,np.inf,np.inf,np.inf]))
    [s, k, l, a1, b1, a2, b2] = params

    predY = f(test[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a1, b1, a2, b2)
    rmse = np.sqrt(np.mean(np.square(predY - test['avg_response_time'])))
    print(test['avg_response_time'])
    print("RMSE -> ", rmse)

    x_max = max(max(df['msg_size']), max(df['concurrent_users']))
    size = max(df['concurrent_users'])
    print(size)

    for msg in [50, 1024, 10240, 102400]:

        x1 = np.full((x_max,), 1)
        x2 = np.full((x_max,), msg)
        x3 = np.arange(0, x_max, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        y = f(new_df[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a1, b1, a2, b2)

        plt.plot(x3, y, label="msg_size="+str(msg))

    plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()

    for scenario in [1,2]:

        x1 = np.full((x_max,), scenario)
        x2 = np.full((x_max,), 50)
        x3 = np.arange(0, x_max, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        y = f(new_df[['scenario', 'msg_size', 'concurrent_users']], s, k, l, a1, b1, a2, b2)

        plt.plot(x3, y, label="scenario="+str(scenario))

    plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()

    # ax = plt.axes(projection='3d')
    # ax.scatter3D(x2, x3, y, c=y)
    # ax.set_xlabel('msg_size')
    # ax.set_ylabel('concurrent_users')
    # ax.set_zlabel('avg_response_time')
    # plt.show()
    # print(df)

    return rmse

def get_average_error():
    error = []
    for i in range(5):
        error.append(create_model())

    print("\n".join([str(e) for e in error]), "\n\n")
    print(np.mean(error))

get_average_error()