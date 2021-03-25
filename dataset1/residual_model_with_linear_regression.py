import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datasets
import models
import linear_regression as linear_regression_model
import numpy as np
import pickle as pkl


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


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
residual_models_predictions = {}

# results_file = open("./results/residual_results.txt", "w")

# load data
print("[INFO] loading data...")
df = datasets.load_data()

def train_models():

    for name, group in df:

        # if name not in test_apis:
        #     continue

        group = datasets.remove_outliers(group)
        print(name, "\n")
        
        group["regression_latency"] = linear_regression_model.predict(name, group["wip"].values.reshape(-1, 1))
        group["residuals"] = group["regression_latency"] - group["latency"]

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        print("[INFO] processing data...")

        #scaling

        trainY = train["residuals"]
        testY = test["residuals"]

        scalerX = MinMaxScaler()
        trainX = scalerX.fit_transform(train["wip"].values.reshape(-1,1))
        testX = scalerX.transform(test["wip"].values.reshape(-1,1))

        # save scaler X
        outfile = open("../../models/api_metrics/new_model_res/_scalars/scalerX" + name.replace("/", "_") + ".pkl", "wb")
        pkl.dump(scalerX, outfile)
        outfile.close()

        model = models.create_residual_model(trainX.shape[1])
        opt = Adam(learning_rate=1e-2, decay=1e-3/200)
        model.compile(loss=root_mean_squared_error, optimizer=opt)

        print("[INFO] training model...")
        history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

        # save model
        model.save('../../models/api_metrics/new_model_res/' + name.replace("/", "_"))

        loss.append(history.history['loss'][-1])
        validation_loss.append(history.history['val_loss'][-1])

        print("[INFO] predicting latency...")
        predY = model.predict(testX)

        # predY = residuals = regression_latency - latency
        # pred_latency = regression_latency  - (regression_latency - latency)

        pred_latency = test["regression_latency"] - predY.flatten()
        rmse = np.sqrt(np.mean(np.square(test["latency"].values - pred_latency)))
        # mae = np.mean(np.abs(test["latency"].values - pred_latency))
        prediction_loss.append(rmse)

        # evaluation using bucket method
        error = []
        pred_error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = scalerX.transform(test_sample["wip"].values.reshape(-1,1))
            # testY = scalerY.transform(test_sample["residuals"].values.reshape(-1,1))
            testY = test_sample["residuals"]

            # predict residual (regression_latency - latency)
            predY = model.predict(testX)

            # residual error
            rmse = np.sqrt(np.mean(np.square(testY.values - predY))) #remove .values for minmax scaler
            # mae = np.mean(np.abs(testY.values - predY))
            error.append(rmse)

            # prediction error (latency)
            # pred_latency = test_sample["regression_latency"] - scalerY.inverse_transform(predY).flatten()
            pred_latency = test_sample["regression_latency"] - predY.flatten()
            rmse = np.sqrt(np.mean(np.square(test_sample["latency"].values - pred_latency)))
            # mae = np.mean(np.abs(test_sample["latency"].values - pred_latency))
            pred_error.append(rmse)

            # print("test sample ", i, " ", test.shape, test_sample.shape, testY.mean(), mae)

        avg_error = np.mean(error)
        print("Residual sample_loss: ", avg_error)
        # sample_loss.append(scalerY.inverse_transform(np.array(avg_error).reshape(-1,1))[0,0])
        sample_loss.append(avg_error)

        avg_error = np.mean(pred_error)
        print("Pred sample_loss: ", avg_error)
        sample_prediction_loss.append(avg_error)


    mean_loss = np.mean(loss)
    mean_val_loss = np.mean(validation_loss)
    mean_prediction_loss = np.mean(prediction_loss)
    mean_sample_loss = np.mean(sample_loss)
    mean_sample_prediction_loss = np.mean(sample_prediction_loss)

    median_loss = np.percentile(loss, 50)
    median_val_loss = np.percentile(validation_loss, 50)
    median_prediction_loss = np.percentile(prediction_loss, 50)
    median_sample_loss = np.percentile(sample_loss, 50)
    median_sample_prediction_loss = np.percentile(sample_prediction_loss, 50)

    percentile_loss = np.percentile(loss, 95)
    percentile_val_loss = np.percentile(validation_loss, 95)
    percentile_prediction_loss = np.percentile(prediction_loss, 95)
    percentile_sample_loss = np.percentile(sample_loss, 95)
    percentile_sample_prediction_loss = np.percentile(sample_prediction_loss, 95)

    print("training loss\n")
    print("\n".join([str(l) for l in loss]), "\n\n")
    print("validation loss\n")
    print("\n".join([str(l) for l in validation_loss]), "\n\n")
    print("bucket sampling\n")
    print("\n".join([str(l) for l in sample_loss]), "\n\n")
    print("prediction error\n")
    print("\n".join([str(l) for l in prediction_loss]), "\n\n")
    print("prediction error (bucket sampling)\n")
    print("\n".join([str(l) for l in sample_prediction_loss]), "\n\n")

    print("Mean loss/val_loss/sample_loss/prediction_loss/sample_predction_loss", mean_loss, mean_val_loss, mean_sample_loss, mean_prediction_loss, mean_sample_prediction_loss)
    print("Median loss/val_loss/sample_loss/prediction_loss//sample_prediction_loss", median_loss, median_val_loss, median_sample_loss, median_prediction_loss, median_sample_prediction_loss)
    print("95th percentile loss/val_loss/sample_loss/prediction_loss//sample_prediction_loss", percentile_loss, percentile_val_loss, percentile_sample_loss, percentile_prediction_loss, percentile_sample_prediction_loss)

  
def evaluate_models():
    for name, group in df:

        # if name == "ballerina/http/Caller#respond":
        #     continue

        # if (name not in test_apis):
        #     continue

        group = datasets.remove_outliers(group)

        group["regression_latency"] = linear_regression_model.predict(name, group["wip"].values.reshape(-1, 1))
        group["residuals"] = group["regression_latency"] - group["latency"]

        infile = open("../../models/api_metrics/new_model_res/_scalars/scalerX" + name.replace("/", "_") + ".pkl", "rb")
        scalerX = pkl.load(infile)
        infile.close()

        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        model = keras.models.load_model('../../models/api_metrics/20_residual_model_with_linear_regression_rmse/' + name.replace("/", "_"), compile=False)

        # preds for ml curve
        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        regression_latency = linear_regression_model.predict(name, x.reshape(-1, 1))

        preds = model.predict(scalerX.transform(x.reshape(-1, 1)))
        # preds = regression_latency - scalerY.inverse_transform(preds).flatten()
        preds = regression_latency - preds.flatten()

        # print(name, preds)

        # preds for dataset
        testX = scalerX.transform(test["wip"].values.reshape(-1,1))
        # testY = scalerY.transform(test["residuals"].values.reshape(-1,1))
        testY = test["residuals"]
        predY = model.predict(testX)

        # predY = d_latency  - (d_latency - latency)
        # pred_latency = test["regression_latency"] - scalerY.inverse_transform(predY).flatten()
        pred_latency = test["regression_latency"] - predY.flatten()
        rmse = np.sqrt(np.mean(np.square(test["latency"].values - pred_latency)))
        # mae = np.mean(np.abs(test["latency"].values - pred_latency))
        prediction_loss.append(rmse)


        # evaluation using bucket method
        error = []
        pred_error = []

        for i in range(5):
            test_sample = datasets.get_test_sample(test)
            testX = scalerX.transform(test_sample["wip"].values.reshape(-1,1))
            # testY = scalerY.transform(test_sample["residuals"].values.reshape(-1,1))
            testY = test_sample["residuals"]

            # predict residual (regression_latency - latency)
            predY = model.predict(testX)

            # residual error
            rmse = np.sqrt(np.mean(np.square(testY.values - predY)))
            # mae = np.mean(np.abs(testY.values - predY))
            error.append(rmse)

            # prediction error (latency)
            # pred_latency = test_sample["regression_latency"] - scalerY.inverse_transform(predY).flatten()
            pred_latency = test_sample["regression_latency"] - predY.flatten()
            rmse = np.sqrt(np.mean(np.square(test_sample["latency"].values - pred_latency)))
            # mae = np.mean(np.abs(test_sample["latency"].values - pred_latency))
            pred_error.append(rmse)


        avg_error = np.mean(error)
        print("Residual sample_loss: ", avg_error)
        sample_loss.append(avg_error)

        avg_error = np.mean(pred_error)
        print("Pred sample_loss: ", avg_error)
        sample_prediction_loss.append(avg_error)


        # plt.yscale("log")
        plt.scatter(group["wip"], group["latency"], label='data')
        plt.scatter(group["wip"], group["residuals"], label='data')
        # plt.scatter(test["wip"], pred_y, label='test data')
        # plt.plot(x, preds, 'r', label='residual line')
        plt.plot(x, regression_latency, 'r', label='domain')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        # plt.show()
        # plt.savefig('../../Plots/residual_actual_domain/' + name.replace("/", "_") + '_loss.png')
        plt.close()

    mean_prediction_loss = np.mean(prediction_loss)
    mean_sample_loss = np.mean(sample_loss)
    mean_sample_prediction_loss = np.mean(sample_prediction_loss)

    median_prediction_loss = np.percentile(prediction_loss, 50)
    median_sample_loss = np.percentile(sample_loss, 50)
    median_sample_prediction_loss = np.percentile(sample_prediction_loss, 50)

    percentile_prediction_loss = np.percentile(prediction_loss, 95)
    percentile_sample_loss = np.percentile(sample_loss, 95)
    percentile_sample_prediction_loss = np.percentile(sample_prediction_loss, 95)

    print("residual error (bucket sampling)\n")
    print("\n".join([str(l) for l in sample_loss]), "\n\n")
    print("prediction error\n")
    print("\n".join([str(l) for l in prediction_loss]), "\n\n")
    print("prediction error (bucket sampling)\n")
    print("\n".join([str(l) for l in sample_prediction_loss]), "\n\n")

    print("Mean sample_loss/prediction_loss/sample_predction_loss", mean_sample_loss, mean_prediction_loss, mean_sample_prediction_loss)
    print("Median sample_loss/prediction_loss/sample_prediction_loss", median_sample_loss, median_prediction_loss, median_sample_prediction_loss)
    print("95th percentile sample_loss/prediction_loss/sample_prediction_loss", percentile_sample_loss, percentile_prediction_loss, percentile_sample_prediction_loss)

    return residual_models_predictions


def get_residual_model_forecasts():

    for name, group in df:

        group = datasets.remove_outliers(group)

        group["regression_latency"] = linear_regression_model.predict(name, group["wip"].values.reshape(-1, 1))
        group["residuals"] = group["regression_latency"] - group["latency"]

        infile = open("../../models/20_residual_model_with_linear_regression_rmse/_scalars/scalerX" + name.replace("/", "_") + ".pkl", "rb")
        scalerX = pkl.load(infile)
        infile.close()

        model = keras.models.load_model('../../models/20_residual_model_with_linear_regression_rmse/' + name.replace("/", "_"), compile=False)

        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        regression_latency = linear_regression_model.predict(name, x.reshape(-1, 1))
        preds = model.predict(scalerX.transform(x.reshape(-1, 1)))
        # preds = regression_latency - scalerY.inverse_transform(preds).flatten()
        preds = regression_latency - preds.flatten()
        residual_models_predictions[name] = preds

        # print(name, preds)

    return residual_models_predictions

train_models()
# evaluate_models()