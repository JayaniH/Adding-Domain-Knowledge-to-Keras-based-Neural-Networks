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

def train_model(i):
    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    domain_model_parameters, _ = domain_model.create_model(df, i)
    
    df['domain_prediction'] = domain_model.predict(df[['scenario', 'msg_size', 'concurrent_users']], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['avg_response_time']

    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    print('[INFO] processing data...')

    #scaling

    trainY = train['residuals']
    testY = test['residuals']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['scenario', 'msg_size', 'concurrent_users']].values.reshape(-1,3))
    testX = scalerX.transform(test[['scenario', 'msg_size', 'concurrent_users']].values.reshape(-1,3))

    # save scaler X
    outfile = open('../../models/api_manager/new_model/_scalars/scalerX.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_residual_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=300, batch_size=4)

    # save model
    model.save('../../models/api_manager/new_model/model')

    # get final loss for residual prediction
    # loss.append(scalerY.inverse_transform(np.array(history.history['loss'][-1]).reshape(-1,3))[0,0])
    # validation_loss.append(scalerY.inverse_transform(np.array(history.history['val_loss'][-1]).reshape(-1,3))[0,0])

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    predY = model.predict(testX)

    domain_error = np.sqrt(np.mean(np.square(test['domain_prediction'] - test['avg_response_time'])))
    print('domain_error', domain_error)
    residual_error = np.sqrt(np.mean(np.square(testY.values - predY.flatten())))
    print('residual_error', residual_error)

    # predY = residuals = domain_prediction - latency
    # pred_response_time = domain_prediction  - (domain_prediction - latency)

    # pred_response_time = test['domain_prediction'] - scalerY.inverse_transform(predY).flatten()
    pred_response_time = test['domain_prediction'] - predY.flatten()
    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'].values - pred_response_time)))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))
    prediction_loss = rmse

    print('\navg_response_time:\n','\n'.join([str(val) for val in test['avg_response_time'].values]))
    print('\npredicted avg_response_time by hybrid model:\n', '\n'.join([str(val) for val in pred_response_time.values]))

    results_df = pd.DataFrame({'scenario': test['scenario'], 'msg_size': test['msg_size'], 'concurrent_users': test['concurrent_users'], 'avg_response_time': test['avg_response_time'], 'domain_prediction': test['domain_prediction'],'residuals': testY , 'residual_prediction': predY.flatten(), 'avg_response_time_prediction': pred_response_time})
    print(results_df)
    results_df.to_csv('../../models/api_manager/new_model/results/case' + str(i+1) + '.csv', sep=",", index= False)

    rms_residuals = np.sqrt(np.mean(np.square(test['residuals'])))
    rms_avg_response_time = np.sqrt(np.mean(np.square(test['avg_response_time'])))

    print('loss/val_loss/domain_loss/residual_loss/prediction_loss/', loss, validation_loss, domain_error, residual_error, prediction_loss)

    print('percentage error residuals/ domain/ hybrid', (residual_error/rms_residuals) * 100 , (domain_error/rms_avg_response_time) * 100, (prediction_loss/rms_avg_response_time) * 100)

    return prediction_loss, pred_response_time

def evaluate_model(i):
    random.seed(i*2)
    seed = random.randint(1,100)
    print('i, seed ', i, seed)

    domain_model_parameters, _ = domain_model.create_model(df, i)

    df['domain_prediction'] = domain_model.predict(df[['scenario', 'msg_size', 'concurrent_users']], domain_model_parameters)
    df['residuals'] = df['domain_prediction'] - df['avg_response_time']

    infile = open('../../models/api_manager/6_residual_model_domainfunc2(test4)/_scalars/scalerX.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    (train, test) = train_test_split(df, test_size=0.3, random_state=seed)

    model = keras.models.load_model('../../models/api_manager/6_residual_model_domainfunc2(test4)/model', compile=False)


    # preds for dataset
    testX = scalerX.transform(test[['scenario', 'msg_size', 'concurrent_users']].values.reshape(-1,3))
    # testY = scalerY.transform(test['residuals'].values.reshape(-1,3))
    testY = test['residuals']
    predY = model.predict(testX)

    domain_error = np.sqrt(np.mean(np.square(test['domain_prediction'] - test['avg_response_time'])))
    print('domain_error', domain_error)
    residual_error = np.sqrt(np.mean(np.square(testY.values - predY.flatten())))
    print('residual_error', residual_error)

    # predY = d_latency  - (d_latency - latency)
    # pred_response_time = test['domain_prediction'] - scalerY.inverse_transform(predY).flatten()
    pred_response_time = test['domain_prediction'] - predY.flatten()
    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'].values - pred_response_time)))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))
    prediction_loss = rmse

    print('\navg_response_time:\n','\n'.join([str(val) for val in test['avg_response_time'].values]))
    print('\npredicted avg_response_time by hybrid model:\n', '\n'.join([str(val) for val in pred_response_time.values]))

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
    plt.savefig('../../Plots/_api_manager/19_residual_model_test4/' + str(i+1) + '_msg_size.png')
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
    plt.savefig('../../Plots/_api_manager/19_residual_model_test4/' + str(i+1) + '_scenario.png')
    plt.close()

    rms_residuals = np.sqrt(np.mean(np.square(test['residuals'])))
    rms_avg_response_time = np.sqrt(np.mean(np.square(test['avg_response_time'])))

    results_df = pd.DataFrame({'scenario': test['scenario'], 'msg_size': test['msg_size'], 'concurrent_users': test['concurrent_users'], 'avg_response_time': test['avg_response_time'], 'domain_prediction': test['domain_prediction'],'residuals': testY , 'residual_prediction': predY.flatten(), 'avg_response_time_prediction': pred_response_time})
    # print(results_df)
    results_df.to_csv('../../models/api_manager/6_residual_model_domainfunc2(test4)/results/case' + str(i+1) + '.csv', sep=",", index= False)

    print('domain_loss/residual_loss/prediction_loss/', domain_error, residual_error, prediction_loss)
    print('percentage error domain/ residuals/ hybrid', (domain_error/rms_avg_response_time) * 100, (residual_error/rms_residuals) * 100, (prediction_loss/rms_avg_response_time) * 100)
    
    return prediction_loss


def get_residual_model_forecasts(i):

    domain_model_parameters, _ = domain_model.create_model(df, i)

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

def cross_validation():
    error = []
    predictions = []
    
    for i in range(5):
        print('\ncase', i+1)
        rmse, preds = train_model(i)
        error.append(rmse)
        predictions.append(preds)
        print('\n'.join([str(p) for p in predictions[i]]), '\n\n')
        print('rmse--->', error[i], '\n')
        print('------------------------')

    print('\n'.join([str(e) for e in error]), '\n\n')
    print('mean error --->', np.mean(error))


def evaluate():
    error = []
    
    for i in range(5):
        print('\ncase', i+1)
        rmse = evaluate_model(i)
        error.append(rmse)
        print('rmse--->', error[i], '\n')
        print('------------------------')

    print('\n'.join([str(e) for e in error]), '\n\n')
    print('mean error --->', np.mean(error))

# train_model()
# evaluate_model(2)
# cross_validation()
evaluate()
