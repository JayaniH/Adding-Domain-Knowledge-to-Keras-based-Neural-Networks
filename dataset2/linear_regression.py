import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import models
import _helpers
import numpy as np
import pandas as pd
import pickle as pkl
import random

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

    model = LinearRegression(normalize=True)
    model.fit(trainX, trainY)
    score = model.score(trainX, trainY)

    print("[RESULT] model score = ", score)

    outfile = open("../../models/api_manager/new_model/" + str(i+1) + ".pkl", "wb")
    pkl.dump(model, outfile)
    outfile.close()

    predY = model.predict(testX)

    rmse = np.sqrt(np.mean(np.square(testY - predY)))
    # mae = np.mean(np.abs(testY - predY))

    print('[RESULT] RMSE = ', rmse)

    return rmse


def evaluate_model(i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    testX = test[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4)
    testY = test['avg_response_time']

    infile = open("../../models/api_manager/new_model/" + str(i+1) + ".pkl", "rb")
    model = pkl.load(infile)
    infile.close()

    predY = model.predict(testX)

    # remove negative preds
    predY = np.maximum(0.0, predY)
    
    rmse = np.sqrt(np.mean(np.square(testY - predY)))
    # rmspe = (np.sqrt(np.mean(np.square((testY - predY) / (testY + EPSILON))))) * 100
    # mae = np.mean(np.abs(testY - predY))
    
    _helpers.print_predictions(testY.values, predY) #.flatten() ?

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
    infile = open("../../models/api_manager/20_linear_regression/" + str(case + 1) + ".pkl", "rb")
    model = pkl.load(infile)
    infile.close()

    predictions = model.predict(x)
    return predictions


# train_with_cross_validation()
# evaluate_with_cross_validation()