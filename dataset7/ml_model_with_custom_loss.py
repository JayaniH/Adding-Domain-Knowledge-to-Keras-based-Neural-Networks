import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import _helpers
import domain_model
import models
import dataset
import numpy as np
import pandas as pd
import pickle as pkl
import random


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def custom_loss(y, y_pred):
    y_true = y[:,0]
    domain_prediction = y[:,1]
    
    # print("custom loss", y_true, y, y_pred, domain_latency)
    # y_true=K.print_tensor(y_true)
    # domain_latency=K.print_tensor(domain_latency)

    return K.sqrt(K.mean(K.square(y_pred - y_true))) +  (0.001 * K.sqrt(K.mean(K.square(domain_prediction - y_pred))))

# load data
print('[INFO] loading data...')
df = pd.read_csv('APIM_Dataset_truncated.csv', sep=',')
df = dataset.remove_outliers1(df)

domain_model_parameters = domain_model.get_parameters(df)

def train_model(train_i, test_i, i):

    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters[i])

    print('[INFO] processing data...')

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    trainY = train[['latency', 'domain_prediction']]
    testY = test[['latency', 'domain_prediction']]

    #scaling

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['concurrent_users', 'name', 'msg_size', 'sleep_time']].values.reshape(-1,4))
    testX = scalerX.transform(test[['concurrent_users', 'name', 'msg_size', 'sleep_time']].values.reshape(-1,4))

    # save scaler X
    outfile = open('../../models/apim/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=custom_loss, optimizer=opt)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=10000, batch_size=4, callbacks=[es])

    # save model
    model.save('../../models/apim/new_model/K' + str(i+1))

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    latency_prediction = model.predict(testX)

    _helpers.print_predictions(test['latency'].values, latency_prediction.flatten())

    print('[RESULT] loss / val_loss', loss, validation_loss)
    rmse, mae, mape = _helpers.get_error(test['latency'].values, latency_prediction.flatten())

    save_predictions_to_csv(i, test, latency_prediction.flatten())

    return rmse, mae, mape


def evaluate_model(train_i, test_i, i):
    
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters[i])

    infile = open('../../models/apim/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    model = keras.models.load_model('../../models/apim/new_model/K' + str(i+1), compile=False)

    # preds for dataset
    testX = scalerX.transform(test[['concurrent_users', 'name', 'msg_size', 'sleep_time']].values.reshape(-1,4))
    testY = test[['latency', 'domain_prediction']]

    print('[INFO] predicting latency...')
    latency_prediction = model.predict(testX)

    _helpers.print_predictions(test['latency'].values, latency_prediction.flatten())

    rmse, mae, mape = _helpers.get_error(test['latency'].values, latency_prediction.flatten())

    save_predictions_to_csv(i, test, latency_prediction.flatten())
    
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


# helper functions
def save_predictions_to_csv(i, df, latency_prediction):
    results_df = pd.DataFrame({'concurrent_users': df['concurrent_users'], 'name': df['name'], 'msg_size': df['msg_size'], 'sleep_time': df['sleep_time'], 'latency': df['latency'], 'domain_prediction': df['domain_prediction'], 'prediction': latency_prediction})
    # print(results_df)
    results_df.to_csv('../../models/apim/new_model/results/K' + str(i+1) + '.csv', sep=",", index= False)


# train_with_cross_validation()
evaluate_with_cross_validation()