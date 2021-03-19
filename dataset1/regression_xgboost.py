from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import pickle as pkl
import xgboost as xgb
import pandas as pd 
import numpy as np
import datasets


df = datasets.load_data()

def train_models():
    prediction_errors = []

    for name, group in df:

        group = datasets.remove_outliers(group)

        X = group["wip"].values.reshape(group["wip"].shape[0],1)
        y = group["latency"]

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=123)

        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

        xg_reg.fit(train_x, train_y)
        outfile = open("../../models/api_metrics/new_model/xgb_" + name.replace("/", "_") + ".pkl", "wb")
        pkl.dump(xg_reg, outfile)
        outfile.close()

        pred_y = xg_reg.predict(test_x)

        rmse = np.sqrt(np.mean(np.square(test_y - pred_y)))
        # mae = np.mean(np.abs(test_y - pred_y))
        prediction_errors.append(rmse)


    mean_prediction_error = np.mean(prediction_errors)
    median_prediction_error = np.percentile(prediction_errors, 50)
    percentile95_prediction_error = np.percentile(prediction_errors, 95)

    print("-------Regression XGBoost--------")
    print("\n".join([str(l) for l in prediction_errors]), "\n\n")

    print("Mean prediction_error", mean_prediction_error)
    print("Median prediction_error", median_prediction_error)
    print("95th percentile prediction_error", percentile95_prediction_error)

    return mean_prediction_error


def evaluate_models():
    prediction_errors = []

    for name, group in df:

        group = datasets.remove_outliers(group)

        X = group["wip"].values.reshape(group["wip"].shape[0],1)
        y = group["latency"]

        data_dmatrix = xgb.DMatrix(data=X, label=y)

        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=123)

        infile = open("../../models/api_metrics/new_model/xgb_" + name.replace("/", "_") + ".pkl", "rb")
        xg_reg = pkl.load(infile)
        infile.close()

        pred_y = xg_reg.predict(test_x)

        rmse = np.sqrt(np.mean(np.square(test_y - pred_y)))
        # mae = np.mean(np.abs(test_y - pred_y))
        prediction_errors.append(rmse)

        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        y = xg_reg.predict(x.reshape(-1, 1))

        plt.scatter(group["wip"], group["latency"], label='data')
        # plt.scatter(test_x["wip"], pred_y, label='test data')
        plt.plot(x, y, 'r', label='ml curve')
        plt.title(name)
        plt.xlabel('wip')
        plt.ylabel('latency')
        plt.legend()
        plt.show()
        # plt.savefig('../../Plots/ml_test_plots_mae_outliers_removed/' + name.replace("/", "_") + '.png')
        plt.close()

    mean_prediction_error = np.mean(prediction_errors)
    median_prediction_error = np.percentile(prediction_errors, 50)
    percentile95_prediction_error = np.percentile(prediction_errors, 95)

    print("-------Regression XGBoost--------")
    print("\n".join([str(l) for l in prediction_errors]), "\n\n")

    print("Mean prediction_error", mean_prediction_error)
    print("Median prediction_error", median_prediction_error)
    print("95th percentile prediction_error", percentile95_prediction_error)

    return mean_prediction_error


def predict(api, x):
    infile = open("../../models/api_metrics/new_model/xgb_" + api.replace("/", "_") + ".pkl", "rb")
    model = pkl.load(infile)
    infile.close()

    predictions = model.predict(x)
    print(predictions)
    return predictions


def get_forecast():
    xgb_predictions = {}
    for name, group in df:

        group = datasets.remove_outliers(group)

        infile = open("../../models/api_metrics/54_xgb/xgb_" + name.replace("/", "_") + ".pkl", "rb")
        model = pkl.load(infile)
        infile.close()

        # predictions for ml curve
        x = np.arange(0, group.wip.max() + 0.1 , 0.01)
        y = model.predict(x.reshape(-1, 1))
        xgb_predictions[name] = y

    return xgb_predictions

# train_models()
# evaluate_models()
# x = np.arange(0, 1000 + 0.1 , 0.01)
# predict('ballerina/http/Client#delete#https://graph.microsoft.com', x.reshape(-1, 1))