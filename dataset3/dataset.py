import pandas as pd

def create_csv(infile, outfile):
    df = pd.read_csv(infile, sep='|')
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(columns=['Unnamed: 0', 'index', 'Unnamed: 6'])
    print(df)
    df.to_csv(outfile, sep=",", index= False)

# create_csv('tpcw_summary1.csv', 'tpcw_summary.csv')

# df = pd.read_csv('tpcw_summary1.csv', sep=',')
# print(df)