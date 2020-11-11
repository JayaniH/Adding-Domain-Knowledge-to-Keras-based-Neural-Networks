
from sklearn.model_selection import train_test_split
import numpy as np
import datasets

parameters = {}

def run():

    loss = []
    EPSILON =  1e-6

    # results_file = open("./results/domain_results.txt", "w")

    print("[INFO] loading data...")
    df = datasets.load_data()

    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

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

        parameters[name] = params

        predY = np.polyval(params, testX)

        # rmspe = (np.sqrt(np.mean(np.square((testY - predY) / testY)))) * 100
        
        # mean absolute percentage error
        error = np.mean(np.abs((testY - predY) / (testY + EPSILON))) * 100
        
        print("Loss: ", error)
        loss.append(error)

        # results_file.write(str(error))
        # results_file.write("\n")

    mean_loss = np.mean(loss)
    percentile_loss = np.percentile(loss, 95)

    print("Mean loss", mean_loss)
    print("95th percentile loss", percentile_loss)

    # results_file.write("\nMean loss %s" % (mean_loss)) 
    # results_file.write("\n95th percentile loss %s" % (percentile_loss)) 


# run()

def predict(api,wip):
    print("[INFO] predicting d_latency...")

    params = parameters[api]
    d_latency = np.polyval(params, wip)

    return d_latency