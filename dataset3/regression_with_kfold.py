import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import models
import numpy as np
import pandas as pd
import pickle as pkl
import random


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# load data
print('[INFO] loading data...')
df = pd.read_csv('tpcw_summary.csv', sep=',')

def train_model(train_i, test_i, i):

    print('[INFO] processing data...')

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    trainY = train['latency']
    testY = test['latency']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3))
    testX = scalerX.transform(test[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3))

    # save scaler X
    outfile = open('../../models/tcpw/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_model(trainX.shape[1])
    opt = Adam(learning_rate=0.01)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

    # save model
    model.save('../../models/tcpw/new_model/K' + str(i+1))

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    pred_response_time = model.predict(testX)

    # print('\nlatency:\n','\n'.join([str(val) for val in test['latency'].values]))

    rmse = np.sqrt(np.mean(np.square(test['latency'].values - pred_response_time.flatten())))
    # mae = np.mean(np.abs(test['latency'].values - pred_response_time))
    prediction_loss = rmse

    print('loss/val_loss/prediction_loss_rmse', loss, validation_loss, prediction_loss)

    return prediction_loss, pred_response_time.flatten()

def evaluate_model(train_i, test_i, i):

    infile = open('../../models/tcpw/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    model = keras.models.load_model('../../models/tcpw/new_model/K' + str(i+1), compile=False)

    # preds for dataset
    testX = scalerX.transform(test[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3))
    # testY = scalerY.transform(test['latency'].values.reshape(-1,3))
    testY = test['latency']
    pred_response_time = model.predict(testX)

    print('\nlatency:\n','\n'.join([str(val) for val in testY.values]))
    print('\npredicted latency:\n', '\n'.join([str(val) for val in pred_response_time.flatten()]))

    results_df = pd.DataFrame({'concurrent_users': test['concurrent_users'], 'cores': test['cores'], 'workload_mix': test['workload_mix'], 'latency': testY, 'prediction': pred_response_time.flatten()})
    print(results_df)
    results_df.to_csv('../../models/tcpw/new_model/results/case' + str(i+1) + '.csv', sep=",", index= False)

    rmse = np.sqrt(np.mean(np.square(testY.values - pred_response_time.flatten())))
    mae = np.mean(np.abs(testY.values - pred_response_time.flatten()))
    mape = np.mean(np.abs((testY.values - pred_response_time.flatten())/testY.values))*100 

    print('rmse/mae/mape', rmse, mae, mape, '\n')
    
    return rmse, mae, mape


def cross_validation():
    prediction_error = []

    print('[INFO] constructing k fold split...')
    kf = KFold(n_splits=10, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('K', str(i+1), '\n')
        prediction_loss, preds = train_model(train_i, test_i, i)
        prediction_error.append(prediction_loss)
        print(preds, '\n')
        i += 1

    print('\n'.join([str(e) for e in prediction_error]), '\n\n')
    print('mean error --->', np.mean(prediction_error))

def evaluate():
    error_rmse = []
    error_mae = []
    error_mape = []
    
    print('[INFO] constructing k fold split...')
    kf = KFold(n_splits=10, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('\nK', i+1)
        rmse, mae, mape = evaluate_model(train_i, test_i, i)
        error_rmse.append(rmse)
        error_mae.append(mae)
        error_mape.append(mape)
        print('rmse--->', rmse, '\n')
        print('mae--->', mae, '\n')
        print('mape--->', mape, '\n')
        print('------------------------')

        i += 1

    print('\n'.join([str(e) for e in error_rmse]), '\n')
    print('mean rmse --->', np.mean(error_rmse), '\n\n')

    print('\n'.join([str(e) for e in error_mae]), '\n')
    print('mean mae --->', np.mean(error_mae), '\n\n')

    print('\n'.join([str(e) for e in error_mape]), '\n')
    print('mean mape --->', np.mean(error_mape), '\n\n')

cross_validation()
# evaluate()