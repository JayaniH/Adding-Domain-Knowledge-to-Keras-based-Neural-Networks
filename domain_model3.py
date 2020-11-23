
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import datasets

test_preds_domain ={}

# defining the USL equation
def USL(n, s, k, l):
	return (1 + s*(n-1) + k*n*(n-1))/l


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
        params, cov = curve_fit(USL, trainX, trainY)
        print("Fitteed Parameters: ", params)

        [s, k, l] = params

        predY = USL(testX, s, k, l)

        #rmspe
        error = (np.sqrt(np.mean(np.square((testY - predY) / (testY + EPSILON))))) * 100
        
        # mean absolute percentage error
        # error = np.mean(np.abs((testY - predY) / (testY + EPSILON))) * 100
        
        print("Loss: ", error)
        loss.append(error)

        x = np.arange(0, group.wip.max() +1 , 0.1)
        y = USL(x, s, k, l)
        test_preds_domain[name] = y

        plt.yscale("log")
        plt.scatter(group.wip, group.latency)
        plt.plot(x, y, 'y', label='domain')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../Plots/domain_plots_log/' + name.replace("/", "_") + '.png')
        plt.close()


    mean_loss = np.mean(loss)
    percentile_loss = np.percentile(loss, 95)

    print("\n".join([str(l) for l in equation_params]), "\n\n")

    print("\n".join([str(l) for l in loss]), "\n\n")

    print("Mean loss", mean_loss)
    print("95th percentile loss", percentile_loss)

    # results_file.write("\nMean loss %s" % (mean_loss)) 
    # results_file.write("\n95th percentile loss %s" % (percentile_loss)) 

def domain_forecast():
    return test_preds_domain

run()