from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 
import pickle as pkl
import pandas as pd 
import numpy as np
import datasets

loss = []
sample_loss = []
EPSILON =  1e-6

df = datasets.load_data()

def train_models():

    for name, group in df:
        # if (name not in high_loss_apis) and (name not in test_apis):
        #     continue

        group = datasets.remove_outliers(group)

        print(name, "\n")

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        print("[INFO] processing data...")

        trainY = train["latency"] 
        testY = test["latency"] 

        trainX = train["wip"].values.reshape(-1,1)
        testX = test["wip"].values.reshape(-1,1)
    
        # scalerx = MinMaxScaler()

        # trainX = scalerx.fit_transform(train["wip"].values.reshape(-1,1))
        # testX = scalerx.transform(test["wip"].values.reshape(-1,1))

        # # save scaler

        # outfile = open("../../models/52_ml_small_sample/_scalars/scaler_" + name.replace("/", "_") + ".pkl", "wb")
        # pkl.dump(scalerx, outfile)
        # outfile.close()

        model = LinearRegression(normalize=True)
        model.fit(trainX, trainY)
        score = model.score(trainX, trainY)

        print("[RESULT] model score = ", score)

        outfile = open("../../models/api_metrics/new_model/" + name.replace("/", "_") + ".pkl", "wb")
        pkl.dump(model, outfile)
        outfile.close()

        preds = model.predict(testX)

        # rmse = np.sqrt(np.mean(np.square(testY - preds)))
        # rmspe = (np.sqrt(np.mean(np.square((testY - preds) / (testY + EPSILON))))) * 100
        mae = np.mean(np.abs(testY - preds))
        loss.append(mae)

        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        y = model.predict(x.reshape(-1, 1))

        # print(name, preds)

        plt.scatter(group["wip"], group["latency"], label='data')
        # plt.scatter(testX["wip"], preds, label='test data')
        plt.plot(x, y, 'r', label='ml curve')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../../Plots/ml_test_plots_mae_outliers_removed/' + name.replace("/", "_") + '_loss.png')
        plt.close()

    mean_loss = np.mean(loss)
    median_loss = np.percentile(loss, 50)
    percentile_loss = np.percentile(loss, 95)

    print("-------Linear Regression--------")
    print("\n".join([str(l) for l in loss]), "\n\n")

    print("Mean loss", mean_loss)
    print("Median loss", median_loss)
    print("95th percentile loss", percentile_loss)

    return mean_loss

def evaluate_models():
    for name, group in df:
        # if (name not in high_loss_apis) and (name not in test_apis):
        #     continue

        group = datasets.remove_outliers(group)

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        print("[INFO] processing data...")

        trainY = train["latency"] 
        testY = test["latency"] 

        trainX = train["wip"].values.reshape(-1,1)
        testX = test["wip"].values.reshape(-1,1)

        infile = open("../../models/api_metrics/18_linear_regression/" + name.replace("/", "_") + ".pkl", "rb")
        model = pkl.load(infile)
        infile.close()

        preds = model.predict(testX)
        
        # rmse = np.sqrt(np.mean(np.square(testY - preds)))
        # rmspe = (np.sqrt(np.mean(np.square((testY - preds) / (testY + EPSILON))))) * 100
        mae = np.mean(np.abs(testY - preds))
        loss.append(mae)

        # evaluation using bucket method
        error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(testX)
            testY = test_sample["latency"] #/ maxLatency

            predY = model.predict(testX)
            mae = np.mean(np.abs(testY.values - predY))
            # rmse = np.sqrt(np.mean(np.square(testY.values - predY)))
            error.append(mae)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(error)
        print("Loss: ", avg_error)
        sample_loss.append(avg_error) # * maxLatency

        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        y = model.predict(x.reshape(-1, 1))

        # print(name, preds)

        plt.scatter(group["wip"], group["latency"], label='data')
        # plt.scatter(testX["wip"], preds, label='test data')
        plt.plot(x, y, 'r', label='ml curve')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../../Plots/ml_test_plots_mae_outliers_removed/' + name.replace("/", "_") + '_loss.png')
        plt.close()

    mean_loss = np.mean(loss)
    median_loss = np.percentile(loss, 50)
    percentile_loss = np.percentile(loss, 95)

    mean_sample_loss = np.mean(sample_loss)
    median_sample_loss = np.percentile(sample_loss, 50)
    percentile_sample_loss = np.percentile(sample_loss, 95)

    print("-------Linear Regression--------")
    print("\n".join([str(l) for l in loss]), "\n\n")
    print("\n".join([str(l) for l in sample_loss]), "\n\n")

    print("Mean loss/sample_loss", mean_loss, mean_sample_loss)
    print("Median loss/sample_loss", median_loss, median_sample_loss)
    print("95th percentile loss/sample_loss", percentile_loss, percentile_sample_loss)

    return mean_loss


def predict(api, x):
    infile = open("../../models/api_metrics/18_linear_regression/" + api.replace("/", "_") + ".pkl", "rb")
    model = pkl.load(infile)
    infile.close()

    predictions = model.predict(x)
    print(predictions)
    return predictions


# train_models()
# evaluate_models()
# x = np.arange(0, 1000 + 0.1 , 0.01)
# predict('ballerina/http/Client#delete#https://graph.microsoft.com', x.reshape(-1, 1))