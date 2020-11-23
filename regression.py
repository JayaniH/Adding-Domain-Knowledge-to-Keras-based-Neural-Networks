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


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 100

scalerx = MinMaxScaler()
scalery = MinMaxScaler()

loss = []
validation_loss = []
high_loss_apis = [
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/Client#get#http://postman-echo.com',
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
    'ballerina/http/Client#post#http://airline-mock-svc:8090',
    'ballerina/http/Client#post#http://car-mock-svc:8090',
    'ballerina/http/HttpClient#forward',
    'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#post',
    'ballerina/http/HttpClient#get',
    'ballerina/http/HttpCachingClient#post'
]
# test_apis = [
#     'ballerina/http/Client#forward#http://13.90.39.240:8602',
#     'ballerina/http/Client#get#https://ap15.salesforce.com',
#     'ballerina/http/Client#patch#https://graph.microsoft.com',
#     'ballerina/http/Client#post#https://login.salesforce.com/services/oauth2/token',
#     'ballerinax/sfdc/SObjectClient#createRecord'
# ]
top_5_preds = {}
test_preds_regression = {}

# load data
print("[INFO] loading data...")
df = datasets.load_data()


def train_model():
    results_file = open("./results/regression_results.txt", "w")

    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

        if name not in high_loss_apis:
            continue

        print(name, "\n")
        
        results_file.write(name)
        results_file.write("\n")

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        print("[INFO] processing data...")

        #scale Y
        maxLatency = train["latency"].max()
        trainY = train["latency"] / maxLatency
        testY = test["latency"] / maxLatency

        # trainY = scalery.fit_transform(train["latency"].values.reshape(-1,1))
        # testY = scalery.fit_transform(test["latency"].values.reshape(-1,1))
    
        # print(train["latency"], trainY.shape, testY.shape)

        # (trainX, testX) = datasets.process_attributes(train, test)
        trainX = scalerx.fit_transform(train["wip"].values.reshape(-1,1))
        testX = scalerx.fit_transform(test["wip"].values.reshape(-1,1))

        # print(train["wip"], trainX.shape, testX.shape)

        model = models.create_model(trainX.shape[1])
        opt = Adam(learning_rate=1e-2, decay=1e-3/200)
        model.compile(loss=root_mean_squared_error, optimizer=opt)

        print("[INFO] training model...")
        history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=500, batch_size=4)

        # save model
        model.save('../models/test_models_rmse/' + name.replace("/", "_"))
        
        # get final loss
        loss.append(history.history['loss'][-1])
        validation_loss.append(history.history['val_loss'][-1])

        # record results
        # print(history.history)


        # results_file.write(str(history.history))
        # results_file.write("\n\n")

        #################################

    print(test_preds_regression)

    mean_loss = np.mean(loss)
    mean_val_loss = np.mean(validation_loss)

    percentile_loss = np.percentile(loss, 95)
    percentile_val_loss = np.percentile(validation_loss, 95)

    print("\n".join([str(l) for l in loss]), "\n\n")
    print("\n".join([str(l) for l in validation_loss]), "\n\n")

    print("Mean loss/val_loss", mean_loss, mean_val_loss)
    print("95th percentile loss/val_loss", percentile_loss, percentile_val_loss)

    # results_file.write("\nMean loss/val_loss %s / %s" % (mean_loss, mean_val_loss)) 
    # results_file.write("\n95th percentile loss/val_loss %s / %s" % (percentile_loss, percentile_val_loss)) 

    results_file.write("\nloss")
    results_file.write(str(loss))
    results_file.write("\nval_loss")
    results_file.write(str(validation_loss))
    results_file.close()


def run_regression():
    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

        if name not in high_loss_apis:
            continue
        
        # print(group.latency.min())

        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        maxLatency = train["latency"].max()

        model = keras.models.load_model('../models/test_models_rmse/' + name.replace("/", "_"), compile=False)

        x = np.arange(0, group.wip.max() +1 , 0.1)
        preds = model.predict(scalerx.fit_transform(x.reshape(-1, 1)))
        preds = preds * maxLatency
        test_preds_regression[name] = preds
        
        # print(preds)
        # results_file.write(str(preds))

        plt.yscale("log")
        plt.scatter(group.wip, group.latency)
        plt.plot(x, preds, 'r', label='regression')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        plt.savefig('../Plots/regression_test_plots_rmse/' + name.replace("/", "_") + '_loss.png')
        plt.close()



def regression_forecast():
    return top_5_preds, test_preds_regression


# train_model()
run_regression()