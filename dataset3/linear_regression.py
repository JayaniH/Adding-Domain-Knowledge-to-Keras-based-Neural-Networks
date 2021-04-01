import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import _helpers
import models
import numpy as np
import pandas as pd
import pickle as pkl
import random

# load data
print('[INFO] loading data...')
df = pd.read_csv('tpcw_summary.csv', sep=',')

def train_model(train_i, test_i, i):

    print('[INFO] processing data...')

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    trainY = train['latency']
    testY = test['latency']

    trainX = train[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3)
    testX = test[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3)

    model = LinearRegression(normalize=True)
    model.fit(trainX, trainY)
    score = model.score(trainX, trainY)

    print("[RESULT] model score = ", score)

    outfile = open("../../models/tpcw/new_model/" + str(i+1) + ".pkl", "wb")
    pkl.dump(model, outfile)
    outfile.close()

    predY = model.predict(testX)
    
    rmse, mae, mape = _helpers.get_error(testY.values, predY)   

    return rmse, mae, mape


def evaluate_model(train_i, test_i, i):

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    infile = open("../../models/tpcw/new_model/" + str(i+1) + ".pkl", "rb")
    model = pkl.load(infile)
    infile.close()

    testX = test[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3)
    testY = test['latency']

    predY = model.predict(testX)

    # remove negative preds
    predY = np.maximum(0.0, predY)

    _helpers.print_predictions(testY.values, predY)

    rmse, mae, mape = _helpers.get_error(testY.values, predY)
    
    return rmse, mae, mape


def train_with_cross_validation():
    errors = {'rmse' : [], 'mae': [], 'mape': []}

    print('[INFO] constructing k fold split...')
    kf = KFold(n_splits=10, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('--------------------------------------------------------------------------------------------------------------------------------')
        print('K', str(i+1), '\n')
        rmse, mae, mape = train_model(train_i, test_i, i)
        errors['rmse'].append(rmse)
        errors['mae'].append(mae)
        errors['mape'].append(mape)
        i += 1

    _helpers.print_errors(errors)

    return errors


def evaluate_with_cross_validation():
    errors = {'rmse' : [], 'mae': [], 'mape': []}
    
    print('[INFO] constructing k fold split...')
    kf = KFold(n_splits=10, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('--------------------------------------------------------------------------------------------------------------------------------')
        print('\nK', i+1)
        rmse, mae, mape = evaluate_model(train_i, test_i, i)
        errors['rmse'].append(rmse)
        errors['mae'].append(mae)
        errors['mape'].append(mape)
        i += 1

    _helpers.print_errors(errors)

    return errors


def predict(i, x):
    infile = open("../../models/tpcw/14_linear_regression/" + str(i + 1) + ".pkl", "rb")
    model = pkl.load(infile)
    infile.close()

    predictions = model.predict(x)
    return predictions



# train_with_cross_validation()
# evaluate_with_cross_validation()