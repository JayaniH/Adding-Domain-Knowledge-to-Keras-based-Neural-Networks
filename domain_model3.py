
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import datasets

test_preds_domain ={}

# USL equation
def USL(n, s, k, l):
	return (1 + s*(n-1) + k*n*(n-1))/l


def run():

    avg_loss = []
    loss = []
    equation_params = []
    EPSILON =  1e-6

    # results_file = open("./results/domain_results.txt", "w")

    print("[INFO] loading data...")
    df = datasets.load_data()

    for name, group in df:

        print("\n", name)

        print("[INFO] removing outliers...")
        group = datasets.remove_outliers(group)

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        trainX = train["wip"]
        trainY = train["latency"]

        print("[INFO] fittinig parameters...")
        params, cov = curve_fit(USL, trainX, trainY, bounds=([0,0,0],[np.inf, 10, np.inf]))
        print("Fitteed Parameters: ", params)

        equation_params.append(params)
        [s, k, l] = params


        # evaluation using bucket method
        error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = test_sample["wip"]
            testY = test_sample["latency"]

            predY = USL(testX, s, k, l)
            mae = np.mean(np.abs(testY - predY))
            error.append(mae)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(error)
        print("Loss: ", avg_error)
        avg_loss.append(avg_error)

        #rmspe
        # error = (np.sqrt(np.mean(np.square((testY - predY) / (testY + EPSILON))))) * 100
        
        # mean absolute percentage error
        # error = np.mean(np.abs((testY - predY) / (testY + EPSILON))) * 100


        # mean absolute error
        # error = np.mean(np.abs(testY - predY))

        # evaluation without bucket sampling (using the whole dataset)
        testX = test["wip"]
        testY = test["latency"]

        predY = USL(testX, s, k, l)
        mae = np.mean(np.abs(testY - predY))
        print("Error with whole test set/avg using bucket method: ", mae, avg_error)
        
        print("Loss: ", mae)
        loss.append(mae)

        x = np.arange(0, group.wip.max() +0.1 , 0.01)
        y = USL(x, s, k, l)
        test_preds_domain[name] = y

        # plt.yscale("log")
        plt.scatter(group.wip, group.latency)
        plt.plot(x, y, 'y', label='domain')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../Plots/domain_model3_new/' + name.replace("/", "_") + '.png')
        plt.close()

    mean_loss = np.mean(loss)
    percentile_loss = np.percentile(loss, 95)

    mean_avg_loss = np.mean(avg_loss)
    percentile_avg_loss = np.percentile(avg_loss, 95)

    print("-------Domain Model--------")
    print("\n".join([str(l) for l in equation_params]), "\n\n")

    print("\n".join([str(l) for l in avg_loss]), "\n\n")
    print("\n".join([str(l) for l in loss]), "\n\n")

    print("Mean loss", mean_loss)
    print("95th percentile loss", percentile_loss)

    print("Mean avg_loss", mean_avg_loss)
    print("95th percentile avg_loss", percentile_avg_loss)

    # results_file.write("\nMean loss %s" % (mean_loss)) 
    # results_file.write("\n95th percentile loss %s" % (percentile_loss)) 

def domain_forecast():
    return test_preds_domain

run()