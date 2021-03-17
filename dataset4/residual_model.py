import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import _helpers
import models
import domain_model
import numpy as np
import pandas as pd
import pickle as pkl
import random


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# load data
print('[INFO] loading data...')
df = pd.read_csv('springboot_db_truncated.csv', sep=',')

domain_model_parameters = domain_model.get_parameters()

def train_model(train_i, test_i, i):
        
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters[i])
    df['residual'] = df['domain_prediction'] - df['latency']

    print('[INFO] processing data...')

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    trainY = train['latency']
    testY = test['latency']

    #scaling

    trainY = train['residual']
    testY = test['residual']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['concurrent_users', 'heap_size', 'collector']].values.reshape(-1,3))
    testX = scalerX.transform(test[['concurrent_users', 'heap_size', 'collector']].values.reshape(-1,3))

    # save scaler X
    outfile = open('../../models/springboot/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_residual_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

    # save model
    model.save('../../models/springboot/new_model/K' + str(i+1))

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting residuals...')
    residual_prediction = model.predict(testX)

    # residual_prediction = residual = domain_prediction - latency
    # latency_prediction = domain_prediction  - (domain_prediction - latency)

    print('[INFO] calculating latency predictions...')
    latency_prediction = test['domain_prediction'] - residual_prediction.flatten()

    _helpers.print_predictions(test['latency'].values, latency_prediction.values)

    print('[RESULT] loss / val_loss', loss, validation_loss)
    rmse, mae, mape = _helpers.get_error(test['latency'].values, latency_prediction)

    save_predictions_to_csv(i, test, residual_prediction.flatten(), latency_prediction)

    return rmse, mae, mape

def evaluate_model(train_i, test_i, i):
    
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters[i])
    df['residual'] = df['domain_prediction'] - df['latency']

    infile = open('../../models/springboot/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    model = keras.models.load_model('../../models/springboot/new_model/K' + str(i+1), compile=False)

    # preds for dataset
    testX = scalerX.transform(test[['concurrent_users', 'heap_size', 'collector']].values.reshape(-1,3))
    testY = test['residual']

    print('[INFO] predicting residuals...')
    residual_prediction = model.predict(testX)

    # residual_prediction = d_latency  - (d_latency - latency)
    # latency_prediction = test['domain_prediction'] - scalerY.inverse_transform(residual_prediction).flatten()

    print('[INFO] calculating latency predictions...')
    latency_prediction = test['domain_prediction'] - residual_prediction.flatten()

    # remove negative preds
    # latency_prediction = np.maximum(0.0, latency_prediction)

    _helpers.print_predictions(test['latency'].values, latency_prediction.values)

    rmse, mae, mape = _helpers.get_error(test['latency'].values, latency_prediction)

    save_predictions_to_csv(i, test, residual_prediction.flatten(), latency_prediction)
    
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
def save_predictions_to_csv(i, df, residual_prediction, latency_prediction):
    results_df = pd.DataFrame({'concurrent_users': df['concurrent_users'], 'heap_zize': df['heap_size'], 'collector': df['collector'], 'latency': df['latency'], 'domain_prediction': df['domain_prediction'],'residual': df['residual'] , 'residual_prediction': residual_prediction, 'prediction': latency_prediction})
    print(results_df)
    results_df.to_csv('../../models/springboot/new_model/results/K' + str(i+1) + '.csv', sep=",", index= False)

train_with_cross_validation()
# evaluate_with_cross_validation()
