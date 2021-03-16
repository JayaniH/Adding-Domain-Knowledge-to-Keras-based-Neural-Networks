import pandas as pd

def truncate_dataset(infile, outfile):
    df = pd.read_csv(infile, sep=",")
    df = df[['heap', 'user', 'collector', 'average_latency']]
    df = df.rename(columns={'heap': 'heap_size', 'user': 'concurrent_users', 'collector': 'collector', 'average_latency': 'latency'})
    df = df.replace({
        'heap_size': {'800m': 800, '1g': 1000, '4g': 4000}, 
        'collector': {'UseSerialGC': 1, 'UseParallelGC': 2, 'UseG1GC': 3, 'UseConcMarkSweepGC': 4}
        })
    print(df)
    df.to_csv(outfile, sep=",", index= False)

def create_csv(infile, outfile):
    df = pd.read_csv(infile, sep='|')
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(columns=['Unnamed: 0', 'index', 'Unnamed: 36'])
    print(df)
    df.to_csv(outfile, sep=",", index= False)

def truncate_dataset2(infile, outfile):
    df = pd.read_csv(infile, sep=",")
    df = df[['use case', 'size', 'heap', 'user', 'collector', 'average_latency']]
    df = df.rename(columns={'use case': 'use_case', 'heap': 'heap_size', 'user': 'concurrent_users', 'collector': 'collector', 'average_latency': 'latency'})
    df = df.replace({
        'use_case': {'Merge': 1, 'Echo': 2},
        'heap_size': {'100m': 100, '200m': 200, '500m': 500, '1g': 1000, '4g': 4000}, 
        'collector': {'UseSerialGC': 1, 'UseParallelGC': 2, 'UseG1GC': 3, 'UseConcMarkSweepGC': 4}
        })
    print(df)
    df.to_csv(outfile, sep=",", index= False)

# truncate_dataset('springboot_db.csv', 'springboot_db_truncated.csv')
# create_csv('springboot_summary.csv', 'springboot_summary.csv')
truncate_dataset2('springboot_summary.csv', 'springboot_summary_truncated.csv')