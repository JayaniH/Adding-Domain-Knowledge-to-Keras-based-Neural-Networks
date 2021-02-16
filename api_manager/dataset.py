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
    # categories 0/1
    # df['scenario'] = df['scenario'].astype('category')
    # df['scenario'] = df['scenario'].cat.codes

    # categories 1/2
    # df['scenario'] = pd.factorize(df['scenario'])[0] + 1

    # one hot encoding
    df['scenario'] = df['scenario'].str.lower()
    one_hot = pd.get_dummies(df['scenario'], prefix='scenario')
    df = pd.concat([df, one_hot], axis=1)
    # df = df.drop(columns=['scenario'])
    print(one_hot)
    
    print(df)
    df.to_csv(file, index= False)

def summarize(file):
    df = pd.read_csv(file, sep=",")
    print('\navg_response_time')
    print('Min', df['avg_response_time'].min())
    print('Max', df['avg_response_time'].max())
    print('Mean', df['avg_response_time'].mean())
    print('Median', df['avg_response_time'].median())
    print('Standard Deviation', df['avg_response_time'].std())

    print('\nconcurrent_users')
    print('Min', df['concurrent_users'].min())
    print('Max', df['concurrent_users'].max())
    print('Mean', df['concurrent_users'].mean())
    print('Median', df['concurrent_users'].median())
    print('Standard Deviation', df['concurrent_users'].std())

    print('\nmsg_size')
    print('Min', df['msg_size'].min())
    print('Max', df['msg_size'].max())
    print('Mean', df['msg_size'].mean())
    print('Median', df['msg_size'].median())
    print('Standard Deviation', df['msg_size'].std())


def group_and_plot(file):
    df = pd.read_csv(file, sep=',')

    for x in ['msg_size', 'concurrent_users']:
        passthrough = df[df['scenario'] == 1]
        transformation = df[df['scenario'] == 2]

        plt.yscale("log")
        plt.scatter(passthrough[x], passthrough['avg_response_time'], label='passthrough')
        plt.scatter(transformation[x], transformation['avg_response_time'], label='transformation')

        a = 'concurrent_users' if x=='msg_size' else 'msg_size'
        for i, txt in enumerate(passthrough[a]):
            plt.annotate(txt, (passthrough[x][i]+1, passthrough['avg_response_time'][i]+1), color='blue')
        for i, txt in enumerate(transformation[a]):
            plt.annotate(txt, (transformation[x][24+i]+1, transformation['avg_response_time'][24+i]+1))

        plt.xlabel(x)
        plt.ylabel('avg_response_time')
        plt.legend()
        plt.show()

    passthrough = df[df['scenario'] == 1]
    transformation = df[df['scenario'] == 2]
    ax = plt.axes(projection='3d')
    ax.scatter3D(passthrough['msg_size'], passthrough['concurrent_users'], passthrough['avg_response_time'], c=passthrough['avg_response_time'])
    ax.scatter3D(transformation['msg_size'], transformation['concurrent_users'], transformation['avg_response_time'], c=transformation['avg_response_time'])
    ax.set_xlabel('msg_size')
    ax.set_ylabel('concurrent_users')
    ax.set_zlabel('avg_response_time')
    plt.show()

def plot_as_categories(file):
    df = pd.read_csv(file, sep=',')
    for msg in [50, 1024, 10240, 102400]:

        df_filtered = df[(df['msg_size'] == msg) & (df['scenario'] == 1)]
        # plt.yscale('log')
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'], label='msg_size='+str(msg))

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Dataset : scenario = passthrough')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    plt.show()
    # plt.savefig('../../Plots/_api_manager/13_domain_model_minimization_regularization_upper_mean_5std_lower_mean_01std/' + str(i+1) + '_msg_size.png')
    plt.close()

    for scenario_id in [1,2]:

        scenario = 'passthrough' if scenario_id == 1 else 'transformation'
       
        df_filtered = df[(df['scenario'] == scenario_id) & (df['msg_size'] == 50)]
        # plt.yscale('log')
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'], label='scenario='+str(scenario))

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Dataset : msg_size = 50')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    plt.show()
    # plt.savefig('../../Plots/_api_manager/13_domain_model_minimization_regularization_upper_mean_5std_lower_mean_01std/' + str(i+1) + '_scenario.png')
    plt.close()

# truncate_dataset('summary.csv', 'summary_truncated.csv')
# categorical_to_numerical('summary_truncated.csv')
# group_and_plot('summary_truncated.csv')
# summarize('summary_truncated.csv')
# plot_as_categories('summary_truncated.csv')
