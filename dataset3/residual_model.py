import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
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
df = pd.read_csv('tpcw_summary.csv', sep=',')

domain_model_parameters = domain_model.get_parameters()

def train_model(train_i, test_i, i):
        
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['latency']

    print('[INFO] processing data...')

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    trainY = train['latency']
    testY = test['latency']

    #scaling

    trainY = train['residuals']
    testY = test['residuals']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3))
    testX = scalerX.transform(test[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3))

    # save scaler X
    outfile = open('../../models/tpcw/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_residual_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

    # save model
    model.save('../../models/tpcw/new_model/K' + str(i+1))

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    predY = model.predict(testX)


    domain_error = np.sqrt(np.mean(np.square(test['domain_prediction'] - test['latency'])))
    print('domain_error', domain_error)
    residual_error = np.sqrt(np.mean(np.square(testY.values - predY.flatten())))
    print('residual_error', residual_error)

    # predY = residuals = domain_prediction - latency
    # pred_response_time = domain_prediction  - (domain_prediction - latency)

    # pred_response_time = test['domain_prediction'] - scalerY.inverse_transform(predY).flatten()
    pred_response_time = test['domain_prediction'] - predY.flatten()
    rmse = np.sqrt(np.mean(np.square(test['latency'].values - pred_response_time)))
    # mae = np.mean(np.abs(test['latency'].values - pred_response_time))
    prediction_loss = rmse

    print('\nlatency:\n','\n'.join([str(val) for val in test['latency'].values]))
    print('\npredicted latency by hybrid model:\n', '\n'.join([str(val) for val in pred_response_time.values]))

    rms_residuals = np.sqrt(np.mean(np.square(test['residuals'])))
    rms_latency = np.sqrt(np.mean(np.square(test['latency'])))

    results_df = pd.DataFrame({'concurrent_users': test['concurrent_users'], 'cores': test['cores'], 'workload_mix': test['workload_mix'], 'latency': test['latency'], 'domain_prediction': test['domain_prediction'],'residuals': testY , 'residual_prediction': predY.flatten(), 'prediction': pred_response_time})
    print(results_df)
    results_df.to_csv('../../models/tpcw/new_model/results/case' + str(i+1) + '.csv', sep=",", index= False)

    print('loss/val_loss/domain_loss/residual_loss/prediction_loss/', loss, validation_loss, domain_error, residual_error, prediction_loss)

    print('percentage error residuals/ domain/ hybrid', (residual_error/rms_residuals) * 100 , (domain_error/rms_latency) * 100, (prediction_loss/rms_latency) * 100)

    return prediction_loss, pred_response_time

def evaluate_model(train_i, test_i, i):
    
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['latency']

    infile = open('../../models/tpcw/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    model = keras.models.load_model('../../models/tpcw/new_model/K' + str(i+1), compile=False)


    # preds for dataset
    testX = scalerX.transform(test[['concurrent_users', 'cores', 'workload_mix']].values.reshape(-1,3))
    # testY = scalerY.transform(test['residuals'].values.reshape(-1,3))
    testY = test['residuals']
    predY = model.predict(testX)

    domain_error = np.sqrt(np.mean(np.square(test['domain_prediction'] - test['latency'])))
    print('domain_error', domain_error)
    residual_error = np.sqrt(np.mean(np.square(testY.values - predY.flatten())))
    print('residual_error', residual_error)

    # predY = d_latency  - (d_latency - latency)
    # pred_response_time = test['domain_prediction'] - scalerY.inverse_transform(predY).flatten()
    pred_response_time = test['domain_prediction'] - predY.flatten()

    # remove negative preds
    # pred_response_time = np.maximum(0.0, pred_response_time)

    rmse = np.sqrt(np.mean(np.square(test['latency'].values - pred_response_time)))
    mae = np.mean(np.abs(test['latency'].values - pred_response_time))
    mape = np.mean(np.abs((test['latency'].values - pred_response_time)/test['latency'].values))*100 

    prediction_loss = rmse

    print('\nlatency:\n','\n'.join([str(val) for val in test['latency'].values]))
    print('\npredicted latency by hybrid model:\n', '\n'.join([str(val) for val in pred_response_time.values]))

    rms_residuals = np.sqrt(np.mean(np.square(test['residuals'])))
    rms_latency = np.sqrt(np.mean(np.square(test['latency'])))

    results_df = pd.DataFrame({'concurrent_users': test['concurrent_users'], 'cores': test['cores'], 'workload_mix': test['workload_mix'], 'latency': test['latency'], 'domain_prediction': test['domain_prediction'],'residuals': testY , 'residual_prediction': predY.flatten(), 'prediction': pred_response_time})
    print(results_df)
    results_df.to_csv('../../models/tpcw/new_model/results/case' + str(i+1) + '.csv', sep=",", index= False)

    print('domain_loss/residual_loss/prediction_loss/', domain_error, residual_error, prediction_loss)
    print('percentage error domain/ residuals/ hybrid', (domain_error/rms_latency) * 100, (residual_error/rms_residuals) * 100, (prediction_loss/rms_latency) * 100)
    
    return rmse, mae, mape

def train_with_cross_validation():
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

def evaluate_with_cross_validation():
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


train_with_cross_validation()
# evaluate_with_cross_validation()
