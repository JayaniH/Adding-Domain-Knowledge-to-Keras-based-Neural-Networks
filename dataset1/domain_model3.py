
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import datasets

# domain model using curve fit


# USL equation
def USL(n, s, k, l):
	return (1 + s*(n-1) + k*n*(n-1))/l

print("[INFO] loading data...")
df = datasets.load_data()

def fit_parameters_and_evaluate():

    parameters = {}
    sample_loss = []
    loss = []
    equation_params = []
    estimate_s = []
    estimate_k = []
    estimate_l = []

    # results_file = open("./results/domain_results.txt", "w")

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
        parameters[name] = params

        equation_params.append(params)
        [s, k, l] = params
        estimate_s.append(s)
        estimate_k.append(k)
        estimate_l.append(l)


        # evaluation using bucket method
        error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = test_sample["wip"]
            testY = test_sample["latency"]

            predY = USL(testX, s, k, l)
            # mae = np.mean(np.abs(testY - predY))
            rmse = np.sqrt(np.mean(np.square(testY - predY)))
            error.append(rmse)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(error)
        print("Loss: ", avg_error)
        sample_loss.append(avg_error)

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
        # mae = np.mean(np.abs(testY - predY))
        rmse = np.sqrt(np.mean(np.square(testY - predY)))
        print("Error with whole test set/avg using bucket method: ", rmse, avg_error)
        
        print("Loss: ", rmse)
        loss.append(rmse)

        x = np.arange(0, group["wip"].max() +0.1 , 0.01)
        y = USL(x, s, k, l)
        # domain_model_predictions[name] = y

        # plt.yscale("log")
        plt.scatter(group["wip"], group["latency"])
        plt.plot(x, y, 'y', label='domain')
        plt.title('[curve_fit]\n'+name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../Plots/29_domain_model3_small_sample/' + name.replace("/", "_") + '.png')
        plt.close()

    mean_loss = np.mean(loss)
    median_loss = np.percentile(loss, 50)
    percentile_loss = np.percentile(loss, 95)

    mean_sample_loss = np.mean(sample_loss)
    median_sample_loss = np.percentile(sample_loss, 50)
    percentile_sample_loss = np.percentile(sample_loss, 95)

    print("-------Domain Model Results--------\n")
    print("Parameter Estimates\n")
    print("\n".join([str(l) for l in equation_params]), "\n\n")

    print("Parameter Estimates - Sigma(s)\n")
    print("\n".join([str(s) for s in estimate_s]), "\n\n")
    print("Parameter Estimates - Kappa(k)\n")
    print("\n".join([str(k) for k in estimate_k]), "\n\n")
    print("Parameter Estimates - Lambda(l)\n")
    print("\n".join([str(l) for l in estimate_l]), "\n\n")

    print("Prediction Error\n")
    print("\n".join([str(l) for l in loss]), "\n\n")
    print("Prediction Error (bucket sampling)\n")
    print("\n".join([str(l) for l in sample_loss]), "\n\n")

    print("Mean loss/sample_loss", mean_loss, mean_sample_loss)
    print("Median loss/sample_loss", median_loss, median_sample_loss)
    print("95th percentile loss/sample_loss", percentile_loss, percentile_sample_loss)

    # results_file.write("\nMean loss %s" % (mean_loss)) 
    # results_file.write("\n95th percentile loss %s" % (percentile_loss)) 

    return parameters

def predict(api, wip, parameters):
    print("[INFO] predicting domain latency for " + api)

    # [s, k, l] = parameters[api]
    [s, k, l] = parameters
    d_latency = USL(wip, s, k, l)

    return d_latency


def get_domain_forecasts():

    domain_model_predictions ={}

    parameters = fit_parameters_and_evaluate()
    for name, group in df:
        group = datasets.remove_outliers(group)
        x = np.arange(0, group["wip"].max() +0.1 , 0.01)
        y = predict(name, x, parameters[name])
        domain_model_predictions[name] = y
        # print(x.shape, y.shape)

    return domain_model_predictions

fit_parameters_and_evaluate()