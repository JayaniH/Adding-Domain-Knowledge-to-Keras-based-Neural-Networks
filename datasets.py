from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np

cs = MinMaxScaler()


def load_data():
    df = pd.read_csv('performance_data_truncated.csv', sep="\t")
    df = df[df.wip < 1500]
    df = df.groupby(by="api_name")

    return df

# load_data()

def remove_outliers(df):
    n = int(max(df["wip"]))+1

    k = 1.4826                 # scale factor for Gaussian distribution

    window_size = n/10
    n_sigma = 4  

    filtered_df = pd.DataFrame()
    outlier_df = pd.DataFrame()
    
    for i in np.arange(window_size,n+2,window_size):
        windowed_df = df[(df["wip"]>(i - window_size)) & (df["wip"]<=(i))]
        if len(windowed_df)>2:
            x0 = np.median(windowed_df["latency"])
            S0 = k * np.median(np.abs(windowed_df["latency"] - x0))         # median absolute deviation
            # S0 = windowed_df["latency"].std()                             # standard deviation
            filtered_window = windowed_df[np.abs(windowed_df["latency"] - x0) < n_sigma * S0]
            outlier_window = pd.concat([windowed_df,filtered_window]).drop_duplicates(keep=False)
            filtered_df = filtered_df.append([filtered_window], ignore_index=True)
            outlier_df = outlier_df.append([outlier_window], ignore_index=True)
            # print (windowed_df.shape, filtered_window.shape, outlier_window.shape)
        else:
            filtered_df = filtered_df.append([windowed_df], ignore_index=True)

    return filtered_df


def get_test_sample(df):
    n = int(max(df["wip"])) + 1
    sample_size = int(len(df)/10) + 1        

    window_size = n/10

    sample_df = pd.DataFrame()

    for i in np.arange(window_size, n + 1, window_size):
        windowed_df = df[(df["wip"] > (i - window_size)) & (df["wip"] <= (i))]
        if len(windowed_df) > 0:
            sample_window = windowed_df.sample(n=sample_size, replace=True)
            sample_df = sample_df.append([sample_window], ignore_index=True)
            # print (windowed_df.shape, sample_window.shape, sample_df.shape)
        # else:
        #     sample_df = sample_df.append([windowed_df], ignore_index=True)

    return sample_df


def process_attributes(train, test):
    continuous = ["wip"]
    # cs = MinMaxScaler()
    trainX = cs.fit_transform(train[continuous])
    testX = cs.transform(test[continuous])

    return (trainX, testX)

def scale_x(x):
    # cs = MinMaxScaler()
    x = cs.fit_transform(x)
    # print(x)
    return x

def scale_y(y):
    # cs = MinMaxScaler()
    y = cs.fit(y)

    return y

def undo_scale_y(y):
    # cs = MinMaxScaler()
    y = cs.inverse_transform(y)

    return y

def test():
    api = 'ballerina/http/Client#get#http://postman-echo.com'
    df = pd.read_csv('performance_data_truncated.csv', sep="\t")
    df = df[df.wip < 1500]
    df = df[df["api_name"] == api]

    df_filtered = remove_outliers(df)
    print ("Filtered---", df.shape, df_filtered.shape)

    # plt.scatter(df["wip"], df["latency"], label='actual data')
    # plt.scatter(df_filtered["wip"], df_filtered["latency"], label='filtered data')
    # plt.title(api)
    # plt.xlabel('wip')
    # plt.ylabel('latency')
    # plt.legend()
    # plt.show()
    # plt.close()

    df_sampled = get_test_sample(df)
    df_sampled = df_sampled.sort_values(by=['latency'])
    print("Sampled---", df.shape, df_sampled.shape, df_sampled["latency"].min())

# test()

################################
# -----------for each API do the following-------

# error = []
# (train, test) = train_test_split(group, test_size=0.3, random_state=42)

# for i in range(5):
#     test_sample = get_test_sample(test)
#     preds = model.predict(test["wip"])
#     mae = np.mean(np.abs(preds - test["latency"]))
#     error.append(mae)

# np.mean(error)

#################################