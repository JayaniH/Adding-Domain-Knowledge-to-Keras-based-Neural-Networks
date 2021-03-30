import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import models
import _helpers
import numpy as np
import pandas as pd
import pickle as pkl
import xgboost as xgb
import random


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# load data
print('[INFO] loading data...')
df = pd.read_csv('summary_truncated.csv', sep=',')

def train_model(i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)
    
    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    print('[INFO] processing data...')

    #scaling

    trainY = train['avg_response_time']
    testY = test['avg_response_time']

    trainX = train[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4)
    testX = test[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4)

    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

    xg_reg.fit(trainX, trainY)
    outfile = open("../../models/api_manager/new_model/xgb_" + str(i) + ".pkl", "wb")
    pkl.dump(xg_reg, outfile)
    outfile.close()

    predY = xg_reg.predict(testX)

    rmse = np.sqrt(np.mean(np.square(testY - predY)))
    # mae = np.mean(np.abs(testY - predY))

    print('[RESULT] RMSE = ', rmse)

    return rmse


def evaluate_model(i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    infile = open("../../models/api_manager/18_xgb/xgb_" + str(i) + ".pkl", "rb")
    xg_reg = pkl.load(infile)
    infile.close()

    # preds for dataset
    testX = test[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4)
    testY = test['avg_response_time']
    predY = xg_reg.predict(testX)
    
    _helpers.print_predictions(testY.values, predY) #.flatten() ?

    rmse = np.sqrt(np.mean(np.square(testY - predY)))
    # mae = np.mean(np.abs(testY - predY))

    print('\n[RESULT] RMSE = ', rmse)
    
    return rmse


def train_with_cross_validation():
    errors = []
    
    for i in range(5):
        print('--------------------------------------')
        print('\ncase', i+1)
        rmse = train_model(i)
        errors.append(rmse)

    mean_error = _helpers.get_average_error(errors)
    return mean_error

def evaluate_with_cross_validation():
    errors = []
    
    for i in range(5):
        print('--------------------------------------')
        print('\ncase', i+1)
        rmse = evaluate_model(i)
        errors.append(rmse)

    mean_error = _helpers.get_average_error(errors)
    return mean_error


def predict(case, x):
    infile = open("../../models/api_manager/18_xgb/xgb_" + str(case) + ".pkl", "rb")
    model = pkl.load(infile)
    infile.close()

    predictions = model.predict(x)
    print(predictions)
    return predictions


# train_with_cross_validation()
# evaluate_with_cross_validation()