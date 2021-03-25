import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import models
import _helpers
# import domain_model
import domain_model_usl as domain_model
import numpy as np
import pandas as pd
import pickle as pkl
import random


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# load data
print('[INFO] loading data...')
df = pd.read_csv('summary_truncated.csv', sep=',')

# fit domain model parameters

def train_model(i):
    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    domain_model_parameters, _ = domain_model.create_model(df, i)
    
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['avg_response_time']

    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    print('[INFO] processing data...')

    #scaling

    trainY = train['residuals']
    testY = test['residuals']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4))
    testX = scalerX.transform(test[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4))

    # save scaler X
    outfile = open('../../models/api_manager/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_residual_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

    # save model
    model.save('../../models/api_manager/new_model/case' + str(i+1))

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    residual_prediction = model.predict(testX)

    # residual_prediction = residuals = domain_prediction - latency
    # response_time_prediction = domain_prediction  - (domain_prediction - latency)
    response_time_prediction = test['domain_prediction'] - residual_prediction.flatten()

    _helpers.print_predictions(test['avg_response_time'].values, response_time_prediction.values)
    
    save_predictions_to_csv(i, test, residual_prediction.flatten(), response_time_prediction)

    print('\n[RESULT] loss / val_loss', loss, validation_loss)
    rmse = _helpers.get_error(test, residual_prediction.flatten(), response_time_prediction)
    
    return rmse

def evaluate_model(i):
    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    domain_model_parameters, _ = domain_model.create_model(df, i)

    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['avg_response_time']

    infile = open('../../models/api_manager/11_residual_model_outlier_removed/_scalars/scalerX_' + str(i+1) +'.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    model = keras.models.load_model('../../models/api_manager/11_residual_model_outlier_removed/case' + str(i+1), compile=False)

    # preds for dataset
    testX = scalerX.transform(test[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4))
    # testY = scalerY.transform(test['residuals'].values.reshape(-1,3))
    testY = test['residuals']
    residual_prediction = model.predict(testX)

    # residual_prediction = d_latency  - (d_latency - latency)
    # response_time_prediction = test['domain_prediction'] - scalerY.inverse_transform(residual_prediction).flatten()
    response_time_prediction = test['domain_prediction'] - residual_prediction.flatten()

    # remove negative preds
    # response_time_prediction = np.maximum(0.0, response_time_prediction)

    _helpers.print_predictions(test['avg_response_time'].values, response_time_prediction.values)

    save_predictions_to_csv(i, test, residual_prediction.flatten(), response_time_prediction)

    rmse = _helpers.get_error(test, residual_prediction.flatten(), response_time_prediction)

    return rmse


def train_with_cross_validation():
    errors = []
    
    for i in range(5):
        print('\n--------------------------------------------------------------------------------------------')
        print('\ncase', i+1)
        rmse = train_model(i)
        errors.append(rmse)

    rmse = _helpers.get_average_error(errors)
    return rmse

def evaluate_with_cross_validation():
    errors = []
    
    for i in range(5):
        print('\n--------------------------------------------------------------------------------------------')
        print('\ncase', i+1)
        rmse = evaluate_model(i)
        errors.append(rmse)

    rmse = _helpers.get_average_error(errors)
    return rmse

# helper functions
def save_predictions_to_csv(i, test, residual_prediction, latency_prediction):
    results_df = pd.DataFrame({'scenario': test['scenario'], 'msg_size': test['msg_size'], 'concurrent_users': test['concurrent_users'], 'avg_response_time': test['avg_response_time'], 'domain_prediction': test['domain_prediction'],'residuals': test['residuals'] , 'residual_prediction': residual_prediction, 'avg_response_time_prediction': latency_prediction})
    print(results_df)
    results_df.to_csv('../../models/api_manager/new_model/results/case' + str(i+1) + '.csv', sep=",", index= False)




# train_with_cross_validation()
evaluate_with_cross_validation()
