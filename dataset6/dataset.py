import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def truncate_dataset(infile, outfile):
    df = pd.read_csv(infile, sep=',')
    df = df[['Name', 'Concurrent Users', 'Message Size (Bytes)', 'Sleep Time (ms)', 'Average (ms)']]
    df = df.rename(columns={'Name': 'name', 'Concurrent Users': 'concurrent_users', 'Message Size (Bytes)': 'msg_size', 'Sleep Time (ms)': 'sleep_time', 'Average (ms)': 'latency'})
    df = df.replace({
        'name': {'passthrough.bal': 1, 'https_passthrough.bal': 2, 'transformation.bal': 3, 'https_transformation.bal': 4, },
        })
    print(df)
    df.to_csv(outfile, sep=',', index= False)

def plot(file):
    df = pd.read_csv(file, sep=',')
    df = remove_outliers1(df)
    print(df.count())
    plt.scatter(df['concurrent_users'], df['latency'])
    plt.show()

def remove_outliers(df):
    n = int(max(df['concurrent_users']))+1

    k = 1.4826                 # scale factor for Gaussian distribution

    window_size = n/10
    n_sigma = 4  

    filtered_df = pd.DataFrame()
    outlier_df = pd.DataFrame()
    
    for i in np.arange(window_size,n+2,window_size):
        windowed_df = df[(df['concurrent_users']>(i - window_size)) & (df['concurrent_users']<=(i))]
        if len(windowed_df)>2:
            x0 = np.mean(windowed_df['latency'])
            # S0 = k * np.median(np.abs(windowed_df['latency'] - x0))         # median absolute deviation
            S0 = windowed_df['latency'].std()                             # standard deviation
            filtered_window = windowed_df[np.abs(windowed_df['latency'] - x0) < n_sigma * S0]
            outlier_window = pd.concat([windowed_df,filtered_window]).drop_duplicates(keep=False)
            filtered_df = filtered_df.append([filtered_window], ignore_index=True)
            outlier_df = outlier_df.append([outlier_window], ignore_index=True)
            # print (windowed_df.shape, filtered_window.shape, outlier_window.shape)
        else:
            filtered_df = filtered_df.append([windowed_df], ignore_index=True)

    return filtered_df

def remove_outliers1(df):
    q25, q75 = np.percentile(df['latency'], 25), np.percentile(df['latency'], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    df = df[(df['latency'] > lower) & (df['latency'] < upper)]

    return df

def remove_outliers2(df):
    q5, q95 = np.percentile(df['latency'], 5), np.percentile(df['latency'], 95)
    df = df[(df['latency'] > q5) & (df['latency'] < q95)]

    return df

# truncate_dataset('Ballerina_Dataset.csv', 'Ballerina_Dataset_truncated.csv')
# plot('Ballerina_Dataset_truncated.csv')