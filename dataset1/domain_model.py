
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import datasets

parameters = {}
dominant_apis = ['ballerina/http/Caller#respond',
'ballerina/http/Client#get#http://postman-echo.com',        #reg
'ballerina/http/Client#get#https://covidapi.info/api/v1',
'ballerina/http/Client#post#http://airline-mock-svc:8090',  #reg
'ballerina/http/Client#post#http://car-mock-svc:8090',      #reg
'ballerina/http/Client#post#http://hotel-mock-svc:8090',
'ballerina/http/HttpCachingClient#forward',                 #reg
'ballerina/http/HttpClient#forward',
'ballerina/http/HttpClient#get',                            #reg
'ballerina/http/HttpClient#post']

# domain model uisng polyfit
def run():

    loss = []
    equation_params = []
    EPSILON =  1e-6

    # results_file = open("./results/domain_results.txt", "w")

    print("[INFO] loading data...")
    df = datasets.load_data()

    for name, group in df:

        print("\n", name)

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        trainX = train["wip"]
        testX = test["wip"]
        
        trainY = train["latency"]
        testY = test["latency"]

        print("[INFO] fittinig parameters...")
        params = np.polyfit(trainX, trainY, 2)
        print("Fitteed Parameters: ", params)

        [a, b, c] = params
        kappa = a /(a + b+  c)
        sigma = (a + b) / (a + b + c)
        lambd = 1 / (a + b + c)
        equation_params.append([kappa, sigma, lambd])

        parameters[name] = params

        predY = np.polyval(params, testX)

        #rmspe
        error = (np.sqrt(np.mean(np.square((testY - predY) / (testY + EPSILON))))) * 100
        
        # mean absolute percentage error
        # error = np.mean(np.abs((testY - predY) / (testY + EPSILON))) * 100
        
        print("Loss: ", error)
        loss.append(error)

        # results_file.write(str(error))
        # results_file.write("\n")

        # data_plot = pd.DataFrame({"api_name": group.api_name, "wip": group.wip, "latency": group.latency, "d_latency": np.polyval(params, group.wip)})

        # if name in dominant_apis:
        #     sns.lineplot(data=data_plot, x="wip", y="latency")
        #     sns.lineplot(data=data_plot, x="wip", y="d_latency")
        #     plt.title(name)
        #     plt.ylim(ymin=0)
        #     plt.xlim(xmin=0)
        #     plt.show()

    mean_loss = np.mean(loss)
    percentile_loss = np.percentile(loss, 95)

    print("\n".join([str(l) for l in equation_params]), "\n\n")

    print("\n".join([str(l) for l in loss]), "\n\n")

    print("Mean loss", mean_loss)
    print("95th percentile loss", percentile_loss)

    # results_file.write("\nMean loss %s" % (mean_loss)) 
    # results_file.write("\n95th percentile loss %s" % (percentile_loss)) 


run()

def predict(api,wip):
    print("[INFO] predicting d_latency...")

    params = parameters[api]
    d_latency = np.polyval(params, wip)

    return d_latency