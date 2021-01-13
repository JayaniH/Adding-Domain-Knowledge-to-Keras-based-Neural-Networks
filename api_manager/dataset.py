import pandas as pd
import matplotlib.pyplot as plt 

def truncate_dataset(infile, outfile):
    df = pd.read_csv(infile, sep=",")
    df = df[['Scenario Name', 'Concurrent Users', 'Message Size (Bytes)', 'Average Response Time (ms)']]
    df = df.rename(columns={'Scenario Name': 'scenario', 'Concurrent Users': 'concurrent_users', 'Message Size (Bytes)': 'msg_size', 'Average Response Time (ms)': 'avg_response_time'})
    print(df)
    df.to_csv(outfile, sep=",", index= False)

def categorical_to_numerical(file):
    df = pd.read_csv(file, sep=",")
    df['scenario'] = df['scenario'].astype('category')
    df['scenario'] = df['scenario'].cat.codes
    print(df)
    df.to_csv(file, index= False)

def group_and_plot(file):
    df = pd.read_csv(file, sep=',')

    for x in ['msg_size', 'concurrent_users']:
        passthrough = df[df['scenario'] == 'Passthrough']
        transformation = df[df['scenario'] == 'Transformation']

        plt.yscale("log")
        plt.scatter(passthrough[x], passthrough['avg_response_time'], label='passthrough')
        plt.scatter(transformation[x], transformation['avg_response_time'], label='transformation')
        plt.xlabel(x)
        plt.ylabel('avg_response_time')
        plt.legend()
        plt.show()

    passthrough = df[df['scenario'] == 'Passthrough']
    transformation = df[df['scenario'] == 'Transformation']
    ax = plt.axes(projection='3d')
    ax.scatter3D(passthrough['msg_size'], passthrough['concurrent_users'], passthrough['avg_response_time'], c=passthrough['avg_response_time'])
    ax.scatter3D(transformation['msg_size'], transformation['concurrent_users'], transformation['avg_response_time'], c=transformation['avg_response_time'])
    ax.set_xlabel('msg_size')
    ax.set_ylabel('concurrent_users')
    ax.set_zlabel('avg_response_time')
    plt.show()

# truncate_dataset('summary.csv', 'summary_truncated.csv')
# categorical_to_numerical('summary_truncated.csv')
group_and_plot('summary_truncated.csv')
