import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datasets
import models
import numpy as np
import pandas as pd
import pickle as pkl


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 100


loss = []
validation_loss = []
prediction_loss = []
sample_loss = []
high_loss_apis = [
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/Client#get#http://postman-echo.com',
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
    'ballerina/http/Client#post#http://airline-mock-svc:8090',
    'ballerina/http/Client#post#http://car-mock-svc:8090',
    'ballerina/http/HttpClient#forward',
    # 'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#post',
    'ballerina/http/HttpClient#get',
    'ballerina/http/HttpCachingClient#post'
]
test_apis = [
    'ballerina/http/Client#get#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://login.salesforce.com/services/oauth2/token',
    'ballerinax/sfdc/QueryClient#getQueryResult',
    'ballerinax/sfdc/SObjectClient#createOpportunity'
]
regression_predictions = {}

# load data
print("[INFO] loading data...")
df = datasets.load_data()


def train_models():
    results_file = open("./results/regression_results.txt", "w")

    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

        # if name not in test_apis:
        #     continue

        group = datasets.remove_outliers(group)

        # maxLatency = group["latency"].max()

        print(name, "\n")
        
        results_file.write(name)
        results_file.write("\n")

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        print("[INFO] processing data...")

        #scale Y
        trainY = train["latency"] #/ maxLatency
        testY = test["latency"] #/ maxLatency
    
        # scale X
        # (trainX, testX) = datasets.process_attributes(train, test)

        scalerx = MinMaxScaler()

        trainX = scalerx.fit_transform(train["wip"].values.reshape(-1,1))
        testX = scalerx.transform(test["wip"].values.reshape(-1,1))

        # save scaler

        outfile = open("../models/52_regression_small_sample/_scalars/scaler_" + name.replace("/", "_") + ".pkl", "wb")
        pkl.dump(scalerx, outfile)
        outfile.close()

        model = models.create_model(trainX.shape[1])
        opt = Adam(learning_rate=1e-2, decay=1e-3/200)
        model.compile(loss=root_mean_squared_error, optimizer=opt)

        print("[INFO] training model...")
        history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

        # save model
        model.save('../models/52_regression_small_sample/' + name.replace("/", "_"))
        
        # get final loss
        loss.append(history.history['loss'][-1]) # * maxLatency
        validation_loss.append(history.history['val_loss'][-1] ) # * maxLatency

        # evaluation with entire test set
        predY = model.predict(testX)
        # mae = np.mean(np.abs(testY.values - predY))
        rmse = np.sqrt(np.mean(np.square(testY.values - predY)))
        prediction_loss.append(rmse) # * maxLatency

        # evaluation using bucket method
        error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = scalerx.transform(test_sample["wip"].values.reshape(-1,1))
            testY = test_sample["latency"] #/ maxLatency

            predY = model.predict(testX)
            # mae = np.mean(np.abs(testY.values - predY))
            rmse = np.sqrt(np.mean(np.square(testY.values - predY)))
            error.append(rmse)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(error)
        print("Loss: ", avg_error)
        sample_loss.append(avg_error) # * maxLatency

        # record results
        # print(history.history)
        # results_file.write(str(history.history))
        # results_file.write("\n\n")

        #################################


    mean_loss = np.mean(loss)
    mean_val_loss = np.mean(validation_loss)
    mean_prediction_loss = np.mean(prediction_loss)
    mean_sample_loss = np.mean(sample_loss)

    median_loss = np.percentile(loss, 50)
    median_val_loss = np.percentile(validation_loss, 50)
    median_prediction_loss = np.percentile(prediction_loss, 50)
    median_sample_loss = np.percentile(sample_loss, 50)

    percentile_loss = np.percentile(loss, 95)
    percentile_val_loss = np.percentile(validation_loss, 95)
    percentile_prediction_loss = np.percentile(prediction_loss, 95)
    percentile_sample_loss = np.percentile(sample_loss, 95)

    print("\n".join([str(l) for l in loss]), "\n\n")
    print("\n".join([str(l) for l in validation_loss]), "\n\n")
    print("\n".join([str(l) for l in prediction_loss]), "\n\n")
    print("\n".join([str(l) for l in sample_loss]), "\n\n")

    print("Mean loss/val_loss/prediction_loss/sample_loss", mean_loss, mean_val_loss, mean_prediction_loss, mean_sample_loss)
    print("Median loss/val_loss/prediction_loss/sample_loss/", median_loss, median_val_loss, median_prediction_loss, median_sample_loss)
    print("95th percentile loss/val_loss/prediction_loss/sample_loss", percentile_loss, percentile_val_loss, percentile_prediction_loss, percentile_sample_loss)

    # results_file.write("\nMean loss/val_loss %s / %s" % (mean_loss, mean_val_loss)) 
    # results_file.write("\n95th percentile loss/val_loss %s / %s" % (percentile_loss, percentile_val_loss)) 

    results_file.write("\nloss")
    results_file.write(str(loss))
    results_file.write("\nval_loss")
    results_file.write(str(validation_loss))
    results_file.close()


def evaluate_models():
    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

        # if (name not in test_apis):
        #     continue

        # print(group.latency.min())

        group = datasets.remove_outliers(group)

        # maxLatency = group["latency"].max()

        infile = open("../models/5_regression_rmse/_scalars/scaler_" + name.replace("/", "_") + ".pkl", "rb")
        scalerx = pkl.load(infile)
        infile.close()

        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        model = keras.models.load_model('../models/5_regression_rmse/' + name.replace("/", "_"), compile=False)

        # preds for regression curve
        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        preds = model.predict(scalerx.transform(x.reshape(-1, 1)))
        preds = preds # * maxLatency

        # print(name, preds)

        # evaluation with entire test set
        testX = scalerx.transform(test["wip"].values.reshape(-1,1))
        testY = test["latency"] # / maxLatency

        predY = model.predict(testX)
        pred_y = predY # * maxLatency
        # mae = np.mean(np.abs(testY.values - predY))
        rmse = np.sqrt(np.mean(np.square(testY.values - predY)))
        prediction_loss.append(rmse) # * maxLatency
        
        # evaluation using bucket method
        error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = scalerx.transform(test_sample["wip"].values.reshape(-1,1))
            testY = test_sample["latency"] # / maxLatency

            predY = model.predict(testX)
            # mae = np.mean(np.abs(testY.values - predY))
            rmse = np.sqrt(np.mean(np.square(testY.values - predY)))
            error.append(rmse)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(error)
        print("Sample Loss: ", avg_error)
        sample_loss.append(avg_error) # maxLatency

        # plt.yscale("log")
        plt.scatter(group.wip, group.latency, label='data')
        plt.scatter(test["wip"], pred_y, label='test data')
        plt.plot(x, preds, 'r', label='regression line')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../Plots/regression_test_plots_mae_outliers_removed/' + name.replace("/", "_") + '_loss.png')
        plt.close()

    mean_prediction_loss = np.mean(prediction_loss)
    mean_sample_loss = np.mean(sample_loss)

    median_prediction_loss = np.percentile(prediction_loss, 50)
    median_sample_loss = np.percentile(sample_loss, 50)

    percentile_prerdiction_loss = np.percentile(prediction_loss, 95)
    percentile_sample_loss = np.percentile(sample_loss, 95)

    print("\n".join([str(l) for l in prediction_loss]), "\n\n")
    print("\n".join([str(l) for l in sample_loss]), "\n\n")

    print("Mean prediction_loss/sample_loss", mean_prediction_loss, mean_sample_loss)
    print("Median loss/prediction_loss/sample_loss", median_prediction_loss, median_sample_loss)
    print("95th percentile prediction_loss/sample_loss", percentile_prerdiction_loss, percentile_sample_loss)



def get_regression_forecasts():
    for name, group in df:

        group = datasets.remove_outliers(group)
        # maxLatency = group["latency"].max()

        infile = open("../models/12_regression_epoch200_batch4_rmse_ouliers_removed_unscaledY/_scalars/scaler_" + name.replace("/", "_") + ".pkl", "rb")
        scalerx = pkl.load(infile)
        infile.close()

        model = keras.models.load_model('../models/12_regression_epoch200_batch4_rmse_ouliers_removed_unscaledY/' + name.replace("/", "_"), compile=False)

        # preds for regression curve
        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        preds = model.predict(scalerx.transform(x.reshape(-1, 1)))
        preds = preds # * maxLatency
        regression_predictions[name] = preds

        # print(name, preds)
    return regression_predictions

train_models()
# evaluate_models()