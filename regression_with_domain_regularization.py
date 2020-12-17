import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datasets
import models
import domain_model3 as domain_model
import numpy as np
import pickle as pkl

def custom_loss(y_true, y):
    y_pred = y[:,0]
    domain_latency = y[:,1]
    
    # print("custom loss", y_true, y, y_pred, domain_latency)
    # y_true=K.print_tensor(y_true)
    # domain_latency=K.print_tensor(domain_latency)

    return K.sqrt(K.mean(K.square(y_pred - y_true))) +  (0.15 * K.sqrt(K.mean(K.square(domain_latency - y_pred))))

loss = []
validation_loss = []
prediction_loss = []
sample_loss = []
sample_prediction_loss = []
EPSILON =  1e-6

test_apis = [
    'ballerina/http/Client#get#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://login.salesforce.com/services/oauth2/token',
    'ballerinax/sfdc/QueryClient#getQueryResult',
    'ballerinax/sfdc/SObjectClient#createOpportunity'
]
regression_with_domain_regularization_predictions = {}

# load data
print("[INFO] loading data...")
df = datasets.load_data()

# fit domain model parameters
domain_model_parameters = domain_model.fit_parameters_and_evaluate()

def train_models():

    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

        # if name not in test_apis:
        #     continue

        group = datasets.remove_outliers(group)
        print(name, "\n")
        
        group["domain_latency"] = domain_model.predict(name, group["wip"], domain_model_parameters[name])

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        print("[INFO] processing data...")

        #scaling
        trainY = train["latency"]
        testY = test["latency"]
        # print(trainY["domain_latency"])

        scalerX = MinMaxScaler()
        trainX = scalerX.fit_transform(train["wip"].values.reshape(-1,1))
        trainX = np.concatenate((trainX, train["domain_latency"].values.reshape(-1,1)), axis=1)
        testX = scalerX.transform(test["wip"].values.reshape(-1,1))
        testX = np.concatenate((testX, test["domain_latency"].values.reshape(-1,1)), axis=1)
        # print(trainX.shape, testX.shape)

        # save scaler X
        outfile = open("../models/24_regression_with_custom_loss_rmse_2_relu/_scalars/scalerX" + name.replace("/", "_") + ".pkl", "wb")
        pkl.dump(scalerX, outfile)
        outfile.close()

        model = models.create_regularization_model(trainX.shape[1])
        opt = Adam(learning_rate=1e-2, decay=1e-3/200)
        model.compile(loss=custom_loss, optimizer=opt)

        print("[INFO] training model...")
        history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

        # save model
        model.save('../models/24_regression_with_custom_loss_rmse_2_relu/' + name.replace("/", "_"))

        loss.append(history.history['loss'][-1])
        validation_loss.append(history.history['val_loss'][-1])

        print("[INFO] predicting latency...")
        preds = model.predict(testX)
        pred_latency = preds[:,0]
        # print(preds[:,1], test["domain_latency"])

        rmse = np.sqrt(np.mean(np.square(test["latency"].values - pred_latency)))
        # mae = np.mean(np.abs(test["latency"].values - pred_latency))
        prediction_loss.append(rmse)

        # evaluation using bucket method
        pred_error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = scalerX.transform(test_sample["wip"].values.reshape(-1,1))
            # testY = scalerY.transform(test_sample["residuals"].values.reshape(-1,1))
            testX = np.concatenate((testX, test_sample["domain_latency"].values.reshape(-1,1)), axis=1)
            testY = test_sample["latency"]

            preds = model.predict(testX)
            pred_latency = preds[:,0]

            # prediction error (latency)
            rmse = np.sqrt(np.mean(np.square(test_sample["latency"].values - pred_latency)))
            # mae = np.mean(np.abs(test_sample["latency"].values - pred_latency))
            pred_error.append(rmse)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(pred_error)
        print("Pred sample_loss: ", avg_error)
        sample_prediction_loss.append(avg_error)


    mean_loss = np.mean(loss)
    mean_val_loss = np.mean(validation_loss)
    mean_prediction_loss = np.mean(prediction_loss)
    mean_sample_prediction_loss = np.mean(sample_prediction_loss)

    median_loss = np.percentile(loss, 50)
    median_val_loss = np.percentile(validation_loss, 50)
    median_prediction_loss = np.percentile(prediction_loss, 50)
    median_sample_prediction_loss = np.percentile(sample_prediction_loss, 50)

    percentile_loss = np.percentile(loss, 95)
    percentile_val_loss = np.percentile(validation_loss, 95)
    percentile_prediction_loss = np.percentile(prediction_loss, 95)
    percentile_sample_prediction_loss = np.percentile(sample_prediction_loss, 95)

    print("\n".join([str(l) for l in loss]), "\n\n")
    print("\n".join([str(l) for l in validation_loss]), "\n\n")
    print("\n".join([str(l) for l in prediction_loss]), "\n\n")
    print("\n".join([str(l) for l in sample_prediction_loss]), "\n\n")

    print("Mean loss/val_loss/prediction_loss/sample_predction_loss", mean_loss, mean_val_loss, mean_prediction_loss, mean_sample_prediction_loss)
    print("Median loss/val_loss/prediction_loss/sample_prediction_loss", median_loss, median_val_loss, median_prediction_loss, median_sample_prediction_loss)
    print("95th percentile loss/val_loss/prediction_loss/sample_prediction_loss", percentile_loss, percentile_val_loss, percentile_prediction_loss, percentile_sample_prediction_loss)


def evaluate_models():
    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

        # if (name not in test_apis):
        #     continue

        group = datasets.remove_outliers(group)

        group["domain_latency"] = domain_model.predict(name, group["wip"], domain_model_parameters[name])

        infile = open("../models/24_regression_with_custom_loss_rmse_2_relu/_scalars/scalerX" + name.replace("/", "_") + ".pkl", "rb")
        scalerX = pkl.load(infile)
        infile.close()

        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        model = keras.models.load_model('../models/24_regression_with_custom_loss_rmse_2_relu/' + name.replace("/", "_"), compile=False)

        # preds for regression curve
        x = np.arange(0, group["wip"].max() + 0.1 , 0.01)
        domain_latency = domain_model.predict(name, x, domain_model_parameters[name])
        xWIP = np.concatenate((scalerX.transform(x.reshape(-1, 1)), domain_latency.reshape(-1,1)), axis=1)
        preds = model.predict(xWIP)
        xPreds = preds[:,0]
        # print(x.shape, xPreds.shape)

        # print(name, preds)

        # preds for dataset
        testX = scalerX.transform(test["wip"].values.reshape(-1,1))
        testX = np.concatenate((testX, test["domain_latency"].values.reshape(-1,1)), axis=1)
        predY = model.predict(testX)[:,0]

        rmse = np.sqrt(np.mean(np.square(test["latency"].values - predY)))
        # mae = np.mean(np.abs(test["latency"].values - pred_latency))
        prediction_loss.append(rmse)


        # evaluation using bucket method
        pred_error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = scalerX.transform(test_sample["wip"].values.reshape(-1,1))
            testX = np.concatenate((testX, test_sample["domain_latency"].values.reshape(-1,1)), axis=1)
            # predict residual (domain_latency - latency)
            pred_latency = model.predict(testX)[:,0]

            # prediction error (latency)
            rmse = np.sqrt(np.mean(np.square(test_sample["latency"].values - pred_latency)))
            # mae = np.mean(np.abs(test_sample["latency"].values - pred_latency))
            pred_error.append(rmse)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(pred_error)
        print("Pred sample_loss: ", avg_error)
        sample_prediction_loss.append(avg_error)


        # plt.yscale("log")
        plt.scatter(group["wip"], group["latency"], label='data')
        plt.scatter(test["wip"], predY, label='test data - predictions')
        plt.scatter(test["wip"], test["latency"], label='test data - actual')
        plt.plot(x, xPreds, label='regression')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../Plots/residual_actual_domain/' + name.replace("/", "_") + '_loss.png')
        plt.close()

    mean_prediction_loss = np.mean(prediction_loss)
    mean_sample_prediction_loss = np.mean(sample_prediction_loss)

    median_prediction_loss = np.percentile(prediction_loss, 50)
    median_sample_prediction_loss = np.percentile(sample_prediction_loss, 50)

    percentile_prediction_loss = np.percentile(prediction_loss, 95)
    percentile_sample_prediction_loss = np.percentile(sample_prediction_loss, 95)

    print("\n".join([str(l) for l in prediction_loss]), "\n\n")
    print("\n".join([str(l) for l in sample_prediction_loss]), "\n\n")

    print("Mean prediction_loss/sample_predction_loss", mean_prediction_loss, mean_sample_prediction_loss)
    print("Median prediction_loss/sample_prediction_loss", median_prediction_loss, median_sample_prediction_loss)
    print("95th percentile prediction_loss/sample_prediction_loss", percentile_prediction_loss, percentile_sample_prediction_loss)


def get_regression_with_domain_regularization_forecasts():

    for name, group in df:

        group = datasets.remove_outliers(group)

        group["domain_latency"] = domain_model.predict(name, group["wip"], domain_model_parameters[name])

        infile = open("../models/24_regression_with_custom_loss_rmse_2_relu/_scalars/scalerX" + name.replace("/", "_") + ".pkl", "rb")
        scalerX = pkl.load(infile)
        infile.close()

        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        model = keras.models.load_model('../models/24_regression_with_custom_loss_rmse_2_relu/' + name.replace("/", "_"), compile=False)

        # preds for regression curve
        x = np.arange(0, group["wip"].max() + 0.1 , 0.01)
        domain_latency = domain_model.predict(name, x, domain_model_parameters[name])
        xWIP = np.concatenate((scalerX.transform(x.reshape(-1, 1)), domain_latency.reshape(-1,1)), axis=1)
        preds = model.predict(xWIP)
        xPreds = preds[:,0]
        regression_with_domain_regularization_predictions[name] = xPreds

    return regression_with_domain_regularization_predictions


# train_models()
# evaluate_models()