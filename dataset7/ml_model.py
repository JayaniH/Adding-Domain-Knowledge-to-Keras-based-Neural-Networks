import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import models
import _helpers
import dataset
import numpy as np
import pandas as pd
import pickle as pkl
import random

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# load data
print('[INFO] loading data...')
df = pd.read_csv('APIM_Dataset_truncated.csv', sep=',')
df = dataset.remove_outliers1(df)

def train_model(train_i, test_i, i):

    print('[INFO] processing data...')

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    trainY = train['latency']
    testY = test['latency']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['concurrent_users', 'name', 'msg_size', 'sleep_time']].values.reshape(-1,4))
    testX = scalerX.transform(test[['concurrent_users', 'name', 'msg_size', 'sleep_time']].values.reshape(-1,4))

    # save scaler X
    outfile = open('../../models/apim/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_model(trainX.shape[1])
    opt = Adam(learning_rate=0.01)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=10000, batch_size=8, callbacks=[es])

    # save model
    model.save('../../models/apim/new_model/K' + str(i+1))

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    latency_prediction = model.predict(testX)

    results_df = pd.DataFrame({'concurrent_users': test['concurrent_users'], 'name': test['name'], 'msg_size': test['msg_size'], 'sleep_time': test['sleep_time'], 'latency': testY, 'prediction': latency_prediction.flatten()})
    # print(results_df)
    results_df.to_csv('../../models/apim/new_model/results/K' + str(i+1) + '.csv', sep=",", index= False)
    
    print('[RESULT] loss / val_loss', loss, validation_loss)
    rmse, mae, mape = _helpers.get_error(testY.values, latency_prediction.flatten())   

    return rmse, mae, mape


def evaluate_model(train_i, test_i, i):

    infile = open('../../models/apim/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    model = keras.models.load_model('../../models/apim/new_model/K' + str(i+1), compile=False)

    # preds for dataset
    testX = scalerX.transform(test[['concurrent_users', 'name', 'msg_size', 'sleep_time']].values.reshape(-1,4))
    testY = test['latency']

    latency_prediction = model.predict(testX)
    # remove negative preds
    latency_prediction = np.maximum(0.0, latency_prediction)

    _helpers.print_predictions(testY.values, latency_prediction.flatten())

    results_df = pd.DataFrame({'concurrent_users': test['concurrent_users'], 'name': test['name'], 'msg_size': test['msg_size'], 'sleep_time': test['sleep_time'], 'latency': testY, 'prediction': latency_prediction.flatten()})
    # print(results_df)
    results_df.to_csv('../../models/apim/new_model/results/K' + str(i+1) + '.csv', sep=",", index= False)

    rmse, mae, mape = _helpers.get_error(testY.values, latency_prediction.flatten())
    
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


train_with_cross_validation()
# evaluate_with_cross_validation()