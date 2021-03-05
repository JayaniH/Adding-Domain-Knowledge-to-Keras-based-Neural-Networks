import pandas as pd

def create_csv(infile, outfile):
    df = pd.read_csv(infile, sep='|')
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(columns=['Unnamed: 0', 'index', 'Unnamed: 6'])
    print(df)
    df.to_csv(outfile, sep=",", index= False)


def create_csv_concurrency(infile, outfile):
    df = pd.read_csv(infile, sep=',')
    df = df = df.groupby(by="concurrent_users")
    concurrency = []
    avg_latency = []
    mid_latency = []
    for name, group in df:
            print(name)
            concurrency.append(name)
            avg_latency.append(group['latency'].mean())
            mid_latency.append(group['latency'].median())

    new_df = pd.DataFrame({'concurrent_users': concurrency, 'avg_latency': avg_latency, 'mid_latency': mid_latency})
    print(new_df)
    new_df.to_csv(outfile, sep=",", index= False)


# create_csv('tpcw_summary1.csv', 'tpcw_summary.csv')

# create_csv_concurrency('tpcw_summary.csv', 'tpcw_concurrency.csv')

df = pd.read_csv('tpcw_concurrency.csv', sep=',')
print(df)
