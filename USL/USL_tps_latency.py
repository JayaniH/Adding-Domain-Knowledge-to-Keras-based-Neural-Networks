from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import datasets

apis_to_skip = [
    'ballerina/email/ImapClient#receiveEmailMessage', #opt params not found
    'ballerina/http/Client#get#http://dummy.restapiexample.com/api/v1', #opt params not found
    'ballerina/http/Client#get#http://dummy.restapiexample.com/api/v1#employees', #opt params not found
    'ballerina/http/Client#post#http://dummy.restapiexample.com/api/v1', #opt params not found'
    'ballerina/http/Client#post#http://dummy.restapiexample.com/api/v1#create', #opt params not found
    'ballerina/http/Client#post#https://www.googleapis.com', #opt params not found
    'ballerina/http/Client#post#https://www.googleapis.com#calendar/v3/calendars/ravinduf@wso2.com/events/watch', #opt params not found
]

latency_params = {}
latency_param_estimates = {'a': [], 'b': [], 'g': []}
tps_params = {}
tps_param_estimates = {'a': [], 'b': [], 'g': []}

def USL_latency(n, a, b, g):
	return (1 + a*(n-1) + b*n*(n-1))/g

def USL_tps(n, a, b, g):
    return (g*n)/(1 + a*(n-1) + b*n*(n-1))

df = pd.read_csv('api_metrics.csv', sep="\t")
df['latency'] = df['latency'].apply(lambda x: x/1000)
df = df.groupby(by="api_name")

def fit_to_latency():

    for name, group in df:

        if name in apis_to_skip:
            continue

        print("\n", name)

        print("[INFO] removing outliers...")
        # group = datasets.remove_outliers(group)

        # print("[INFO] constructing training/testing split...")
        # (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        # trainX = train["wip"]
        # trainY = train["latency"]

        print("[INFO] fittinig parameters...")
        params, cov = curve_fit(USL_latency, group["wip"], group["latency"], p0=[0.5,0.5,0.5], bounds=([0,0,0],[1, 1, np.inf]))

        print("Fitteed Parameters: ", params)
        latency_params[name] = params
        a, b, g = params
        latency_param_estimates['a'].append(a)
        latency_param_estimates['b'].append(b)
        latency_param_estimates['g'].append(g)

    print_parameters(latency_param_estimates, 'latency')
    return latency_params

def fit_to_tps():

    for name, group in df:

        if name in apis_to_skip:
            continue

        print("\n", name)

        print("[INFO] removing outliers...")
        # group = datasets.remove_outliers(group)

        # print("[INFO] constructing training/testing split...")
        # (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        # trainX = train["wip"]
        # trainY = train["latency"]

        print("[INFO] fittinig parameters...")
        params, cov = curve_fit(USL_tps, group["wip"], group["latency"], p0=[0.5,0.5,0.5], bounds=([0,0,0],[1, 1, np.inf]))

        print("Fitteed Parameters: ", params)
        tps_params[name] = params
        a, b, g = params
        tps_param_estimates['a'].append(a)
        tps_param_estimates['b'].append(b)
        tps_param_estimates['g'].append(g)

    print_parameters(tps_param_estimates, 'tps')
    return tps_params

def plot_latency(latency_params, tps_params):
    for name, group in df:

        if name in apis_to_skip:
            continue

        a1, b1, g1 = latency_params[name]
        a2, b2, g2 = tps_params[name]
        x = np.arange(0, group["wip"].max() +0.1 , 0.01)
        y1 = USL_latency(x, a1, b1, g1)
        y2 = USL_latency(x, a2, b2, g2)

        # plt.yscale("log")
        plt.scatter(group["wip"], group["latency"])
        plt.plot(x, y1, label='latency eq')
        plt.plot(x, y2, label='tps eq')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        plt.savefig('../../Plots/usl/latency/' + name.replace("/", "_") + '.png')
        plt.close()

def plot_tps(latency_params, tps_params):
    for name, group in df:

        if name in apis_to_skip:
            continue

        a1, b1, g1 = tps_params[name]
        a2, b2, g2 = latency_params[name]
        x = np.arange(0, group["wip"].max() +0.1 , 0.01)
        y1 = USL_tps(x, a1, b1, g1)
        y2 = USL_latency(x, a2, b2, g2)

        # plt.yscale("log")
        plt.scatter(group["wip"], group["throughput"])
        plt.plot(x, y1, label='tps eq')
        plt.plot(x, y2, label='latency eq')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('throughput')
        plt.legend()
        # plt.show()
        plt.savefig('../../Plots/usl/tps/' + name.replace("/", "_") + '.png')
        plt.close()

def create_plots():
    latency_params = fit_to_latency()
    tps_params = fit_to_tps()

    plot_latency(latency_params, tps_params)
    plot_tps(latency_params, tps_params)

def print_parameters(param_estimates, eq):
    print('----------------------')
    print("Parameter Estimates\n")
    print("alpha(a)\n", "\n".join([str(s) for s in param_estimates['a']]), "\n\n")
    print("beta(b)\n", "\n".join([str(k) for k in param_estimates['b']]), "\n\n")
    print("gamma(g)\n", "\n".join([str(l) for l in param_estimates['g']]), "\n\n")

    results_df = pd.DataFrame({'alpha': param_estimates['a'], 'beta': param_estimates['b'], 'gamma': param_estimates['g']})
    # print(results_df)
    results_df.to_csv('../../results/usl_' + str(eq) + '.csv', sep=",", index= False)

# fit_to_latency()
# fit_to_tps()
# plot_latency()
# plot_tps()

create_plots()