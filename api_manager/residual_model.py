import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import models
import domain_model
import numpy as np
import pandas as pd
import pickle as pkl
import random


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 100

# results_file = open('./results/residual_results.txt', 'w')

# load data
print('[INFO] loading data...')
df = pd.read_csv('summary_truncated.csv', sep=',')

# fit domain model parameters
domain_model_parameters, _ = domain_model.create_model(df, 2)

def train_model():
    
    df['domain_prediction'] = domain_model.predict(df[['scenario', 'msg_size', 'concurrent_users']], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['avg_response_time']

    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=74)

    print('[INFO] processing data...')

    #scaling

    trainY = train['residuals']
    testY = test['residuals']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['scenario', 'msg_size', 'concurrent_users']].values.reshape(-1,3))
    testX = scalerX.transform(test[['scenario', 'msg_size', 'concurrent_users']].values.reshape(-1,3))

    # save scaler X
    outfile = open('../../models/api_manager/4_residual_model/_scalars/scalerX.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_residual_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

    # save model
    model.save('../../models/api_manager/4_residual_model/model')

    # get final loss for residual prediction
    # loss.append(scalerY.inverse_transform(np.array(history.history['loss'][-1]).reshape(-1,3))[0,0])
    # validation_loss.append(scalerY.inverse_transform(np.array(history.history['val_loss'][-1]).reshape(-1,3))[0,0])

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    predY = model.predict(testX)

    # predY = residuals = domain_prediction - latency
    # pred_response_time = domain_prediction  - (domain_prediction - latency)

    # pred_response_time = test['domain_prediction'] - scalerY.inverse_transform(predY).flatten()
    pred_response_time = test['domain_prediction'] - predY.flatten()
    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'].values - pred_response_time)))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))
    prediction_loss = rmse

    # # evaluation using bucket method
    # error = []
    # pred_error = []

    # for i in range(5):
    #     test_sample = datasets.get_test_sample(test)
    #     testX = scalerX.transform(test_sample[['scenario', 'msg_size', 'concurrent_users']].values.reshape(-1,3))
    #     # testY = scalerY.transform(test_sample['residuals'].values.reshape(-1,3))
    #     testY = test_sample['residuals']

    #     # predict residual (domain_prediction - latency)
    #     predY = model.predict(testX)

    #     # residual error
    #     # mae = np.mean(np.abs(testY.values - predY))
    #     rmse = np.sqrt(np.mean(np.square(testY.values - predY))) #remove .values for minmax scaler
    #     # mae = np.mean(np.abs(testY.values - predY))
    #     error.append(rmse)

    #     # prediction error (latency)
    #     # pred_response_time = test_sample['domain_prediction'] - scalerY.inverse_transform(predY).flatten()
    #     pred_response_time = test_sample['domain_prediction'] - predY.flatten()
    #     rmse = np.sqrt(np.mean(np.square(test_sample['avg_response_time'].values - pred_response_time)))
    #     # mae = np.mean(np.abs(test_sample['avg_response_time'].values - pred_response_time))
    #     pred_error.append(rmse)

    #     # print('test sample ', i, ' ', test.shape, test_sample.shape, testY.mean(), mae)

    # avg_error = np.mean(error)
    # print('Residual sample_loss: ', avg_error)
    # # sample_loss.append(scalerY.inverse_transform(np.array(avg_error).reshape(-1,3))[0,0])
    # sample_loss = avg_error

    # avg_error = np.mean(pred_error)
    # print('Prediction sample_loss: ', avg_error)
    # sample_prediction_loss = avg_error


    print('loss/val_loss/prediction_loss/sample_loss/sample_predction_loss', loss, validation_loss, prediction_loss)


def evaluate_model():

    df['domain_prediction'] = domain_model.predict(df[['scenario', 'msg_size', 'concurrent_users']], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['avg_response_time']

    infile = open('../../models/api_manager/4_residual_model/_scalars/scalerX.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    (train, test) = train_test_split(df, test_size=0.3, random_state=74)

    model = keras.models.load_model('../../models/api_manager/4_residual_model/model', compile=False)


    # preds for dataset
    testX = scalerX.transform(test[['scenario', 'msg_size', 'concurrent_users']].values.reshape(-1,3))
    # testY = scalerY.transform(test['residuals'].values.reshape(-1,3))
    testY = test['residuals']
    predY = model.predict(testX)

    # predY = d_latency  - (d_latency - latency)
    # pred_response_time = test['domain_prediction'] - scalerY.inverse_transform(predY).flatten()
    pred_response_time = test['domain_prediction'] - predY.flatten()
    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'].values - pred_response_time)))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))
    prediction_loss = rmse

    print('\navg_response_time:\n','\n'.join([str(val) for val in test['avg_response_time'].values]))
    print('\npredicted avg_response_time:\n', '\n'.join([str(val) for val in pred_response_time.values]))


    for msg in [50, 1024, 10240, 102400]:

        x1 = np.full((1000,), 1)
        x2 = np.full((1000,), msg)
        x3 = np.arange(0, 1000, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        domain_prediction = domain_model.predict(new_df, domain_model_parameters)
        y = model.predict(scalerX.transform(new_df.values.reshape(-1,3)))
        # y = domain_prediction - scalerY.inverse_transform(y).flatten()
        y = domain_prediction - y.flatten()


        df_filtered = df[(df['msg_size'] == msg) & (df['scenario'] == 1)]
        # plt.yscale('log')
        plt.plot(x3, y, label='msg_size='+str(msg))
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Residual Model : scenario = passthrough')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    plt.savefig('../../Plots/_api_manager/7_residual_model/msg_size.png')
    plt.close()

    for scenario_id in [1,2]:

        scenario = 'passthrough' if scenario_id == 1 else 'transformation'
        x1 = np.full((1000,), scenario_id)
        x2 = np.full((1000,), 50)
        x3 = np.arange(0, 1000, 1)
        new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
        domain_prediction = domain_model.predict(new_df, domain_model_parameters)
        y = model.predict(scalerX.transform(new_df.values.reshape(-1,3)))
        # y = domain_prediction - scalerY.inverse_transform(y).flatten()
        y = domain_prediction - y.flatten()


        df_filtered = df[(df['scenario'] == scenario_id) & (df['msg_size'] == 50)]
        # plt.yscale('log')
        plt.plot(x3, y, label='scenario='+str(scenario))
        plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    plt.title('Residual Model : msg_size = 50')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    plt.savefig('../../Plots/_api_manager/7_residual_model/scenario.png')
    plt.close()

    print('prediction_loss/sample_loss/sample_predction_loss', prediction_loss)
    
    return prediction_loss


def get_residual_model_forecasts():

    df['domain_prediction'] = domain_model.predict(df[['scenario', 'msg_size', 'concurrent_users']], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['avg_response_time']

    infile = open('../../models/api_manager/4_residual_model/_scalars/scalerX.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    model = keras.models.load_model('../../models/api_manager/4_residual_model/model', compile=False)

    x1 = np.full((1000,), 1)
    x2 = np.full((1000,), 50)
    x3 = np.arange(0, 1000, 1)
    new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
    domain_prediction = domain_model.predict(new_df, domain_model_parameters)
    y = model.predict(scalerX.transform(new_df.values.reshape(-1,3)))
    # y = domain_prediction - scalerY.inverse_transform(y).flatten()
    y = domain_prediction - y.flatten()

    return y

# train_model()
evaluate_model()
# print(get_residual_model_forecasts())