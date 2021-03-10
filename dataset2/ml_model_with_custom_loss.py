import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import domain_model_usl as domain_model
import models
import numpy as np
import pandas as pd
import pickle as pkl
import random


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 100

def custom_loss(y, y_pred):
    y_true = y[:,0]
    domain_prediction = y[:,1]
    
    # print("custom loss", y_true, y, y_pred, domain_latency)
    # y_true=K.print_tensor(y_true)
    # domain_latency=K.print_tensor(domain_latency)

    return K.sqrt(K.mean(K.square(y_pred - y_true))) +  (1 * K.sqrt(K.mean(K.square(domain_prediction - y_pred))))

# results_file = open('./results/residual_results.txt', 'w')

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

    # get final loss for residual prediction
    # loss.append(scalerY.inverse_transform(np.array(history.history['loss'][-1]).reshape(-1,3))[0,0])
    # validation_loss.append(scalerY.inverse_transform(np.array(history.history['val_loss'][-1]).reshape(-1,3))[0,0])

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    pred_response_time = model.predict(testX)

    # print('\navg_response_time:\n','\n'.join([str(val) for val in test['avg_response_time'].values]))

    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'].values - pred_response_time.flatten())))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))
    prediction_loss = rmse

    print('loss/val_loss/prediction_loss/sample_loss/sample_predction_loss', loss, validation_loss, prediction_loss)

    return prediction_loss, pred_response_time.flatten()


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
    # testY = scalerY.transform(test['avg_response_time'].values.reshape(-1,3))
    testY = test[['avg_response_time', 'domain_prediction']]
    pred_response_time = model.predict(testX)

    print('\navg_response_time:\n','\n'.join([str(val) for val in test['avg_response_time'].values]))
    print('\npredicted avg_response_time by ml model with custom loss:\n', '\n'.join([str(val) for val in pred_response_time.flatten()]))

    results_df = pd.DataFrame({'scenario': test['scenario'], 'msg_size': test['msg_size'], 'concurrent_users': test['concurrent_users'], 'avg_response_time': test['avg_response_time'], 'prediction': pred_response_time.flatten()})
    print(results_df)
    results_df.to_csv('../../models/api_manager/new_model/results/case' + str(i+1) + '.csv', sep=",", index= False)

    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'] - pred_response_time.flatten())))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))
    prediction_loss = rmse

    # forecasting

    # for msg in [50, 1024, 10240, 102400]:

    #     x1 = np.full((1000,), 1)
    #     x2 = np.full((1000,), msg)
    #     x3 = np.arange(0, 1000, 1)
    #     new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
    #     y = model.predict(scalerX.transform(new_df.values.reshape(-1,3)))
    #     y = y.flatten()


    #     df_filtered = df[(df['msg_size'] == msg) & (df['scenario'] == 1)]
    #     # plt.yscale('log')
    #     plt.plot(x3, y, label='msg_size='+str(msg))
    #     plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    # plt.title('ML Model : scenario = passthrough')
    # plt.xlabel('concurrent_users')
    # plt.ylabel('avg_response_time')
    # plt.legend()
    # # plt.show()
    # # plt.savefig('../../Plots/_api_manager/6_regression/msg_size.png')
    # plt.close()

    # for scenario_id in [1,2]:

    #     scenario = 'passthrough' if scenario_id == 1 else 'transformation'
    #     x1 = np.full((1000,), scenario_id)
    #     x2 = np.full((1000,), 50)
    #     x3 = np.arange(0, 1000, 1)
    #     new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
    #     y = model.predict(scalerX.transform(new_df.values.reshape(-1,3)))
    #     y = y.flatten()


    #     df_filtered = df[(df['scenario'] == scenario_id) & (df['msg_size'] == 50)]
    #     # plt.yscale('log')
    #     plt.plot(x3, y, label='scenario='+str(scenario))
    #     plt.scatter(df_filtered['concurrent_users'], df_filtered['avg_response_time'])

    # # plt.scatter(df['concurrent_users'], df['avg_response_time'])
    # plt.title('ML Model : msg_size = 50')
    # plt.xlabel('concurrent_users')
    # plt.ylabel('avg_response_time')
    # plt.legend()
    # # plt.show()
    # # plt.savefig('../../Plots/_api_manager/6_regression/scenario.png')
    # plt.close()

    print('prediction_loss/sample_loss/sample_predction_loss', prediction_loss, '\n')
    
    return prediction_loss


def get_rregression_model_forecasts():

    infile = open('../../models/api_manager/3_regression_relu/_scalars/scalerX.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    model = keras.models.load_model('../../models/api_manager/3_regression_relu/model', compile=False)

    x1 = np.full((1000,), 1)
    x2 = np.full((1000,), 50)
    x3 = np.arange(0, 1000, 1)
    new_df = pd.DataFrame({'scenario': x1, 'msg_size': x2, 'concurrent_users': x3})
    y = model.predict(scalerX.transform(new_df.values.reshape(-1,3)))
    y = y.flatten()

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
# evaluate_model()
# cross_validation()
evaluate()