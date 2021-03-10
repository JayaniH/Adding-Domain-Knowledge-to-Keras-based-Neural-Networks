import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import _helpers
import domain_model_usl as domain_model
import models
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

    return K.sqrt(K.mean(K.square(y_pred - y_true))) +  (1 * K.sqrt(K.mean(K.square(domain_prediction - y_pred))))

# load data
print('[INFO] loading data...')
df = pd.read_csv('summary_truncated.csv', sep=',')

def train_model(i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    domain_model_parameters, _ = domain_model.create_model(df, i)
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters)
    
    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    print('[INFO] processing data...')

    #scaling

    trainY = train[['avg_response_time', 'domain_prediction']]
    testY = test[['avg_response_time', 'domain_prediction']]

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4))
    testX = scalerX.transform(test[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4))

    # save scaler X
    outfile = open('../../models/api_manager/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=custom_loss, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

    # save model
    model.save('../../models/api_manager/new_model/case' + str(i+1))

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    pred_response_time = model.predict(testX)

    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'].values - pred_response_time.flatten())))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))

    print('\n[RESULT] loss / val_loss', loss, validation_loss)
    print('[RESULT] RMSE = ', rmse)

    return rmse


def evaluate_model(i):

    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    domain_model_parameters, _ = domain_model.create_model(df, i)
    df['domain_prediction'] = domain_model.predict(df['concurrent_users'], domain_model_parameters)

    infile = open('../../models/api_manager/new_model/_scalars/scalerX_' + str(i+1) +'.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    model = keras.models.load_model('../../models/api_manager/new_model/case' + str(i+1), compile=False)


    # preds for dataset
    testX = scalerX.transform(test[['scenario_passthrough', 'scenario_transformation', 'msg_size', 'concurrent_users']].values.reshape(-1,4))
    testY = test[['avg_response_time', 'domain_prediction']]
    pred_response_time = model.predict(testX)

    _helpers.print_predictions(test['avg_response_time'].values, pred_response_time.flatten())

    save_predictions_to_csv(i,test, pred_response_time.flatten())

    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'] - pred_response_time.flatten())))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))

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

# helper functions
def save_predictions_to_csv(i, test, latency_prediction):
    results_df = pd.DataFrame({'scenario': test['scenario'], 'msg_size': test['msg_size'], 'concurrent_users': test['concurrent_users'], 'avg_response_time': test['avg_response_time'], 'prediction': latency_prediction})
    print(results_df)
    results_df.to_csv('../../models/api_manager/new_model/results/case' + str(i+1) + '.csv', sep=",", index= False)


# train_with_cross_validation()
evaluate_with_cross_validation()