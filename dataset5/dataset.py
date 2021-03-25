import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def truncate_dataset(infile, outfile):
    df = pd.read_csv(infile, sep=',')
    df = df[['use case', 'size', 'heap', 'user', 'collector', 'average_latency']]
    df = df.rename(columns={'use case': 'use_case', 'heap': 'heap_size', 'user': 'concurrent_users', 'collector': 'collector', 'average_latency': 'latency'})
    df = df.replace({
        'use_case': {'Merge': 1, 'Echo': 2},
        'heap_size': {'100m': 100, '200m': 200, '500m': 500, '1g': 1000, '4g': 4000}, 
        'collector': {'UseSerialGC': 1, 'UseParallelGC': 2, 'UseG1GC': 3, 'UseConcMarkSweepGC': 4}
        })
    print(df)
    df.to_csv(outfile, sep=',', index= False)

def plot(file):
    df = pd.read_csv(file, sep=',')
    df = remove_outliers(df)
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
            x0 = np.median(windowed_df['latency'])
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

# truncate_dataset('springboot_summary.csv', 'springboot_summary_truncated.csv')
# plot('springboot_summary_truncated.csv')