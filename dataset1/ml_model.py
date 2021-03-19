import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datasets
import models
import numpy as np
import pandas as pd
import pickle as pkl


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 100

test_apis = [
    'ballerina/http/Client#get#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://login.salesforce.com/services/oauth2/token',
    'ballerinax/sfdc/QueryClient#getQueryResult',
    'ballerinax/sfdc/SObjectClient#createOpportunity'
]
api = [
    'ballerina/http/Client#forward#http://13.90.39.240:8601',
    'ballerina/http/Client#forward#http://13.90.39.240:8602',
    'ballerina/http/Client#forward#http://13.90.39.240:8688',
    'ballerina/http/Client#forward#http://13.90.39.240:8702'
]
ml_predictions = {}

# load data
print('[INFO] loading data...')
df = datasets.load_data()


def train_model(api, df):

    df = datasets.remove_outliers(df)

    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    train_y = train['latency'] 
    test_y = test['latency'] 

    scalerx = MinMaxScaler()

    train_x = scalerx.fit_transform(train['wip'].values.reshape(-1,1))
    test_x = scalerx.transform(test['wip'].values.reshape(-1,1))

    # save scaler
    outfile = open('../../models/api_metrics/new_model/_scalars/scaler_' + api.replace('/', '_') + '.pkl', 'wb')
    pkl.dump(scalerx, outfile)
    outfile.close()

    model = models.create_model(train_x.shape[1])
    opt = Adam(learning_rate=1e-3)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=200, batch_size=4)

    # save model
    model.save('../../models/api_metrics/new_model/' + api.replace('/', '_'))
    
    training_loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    pred_y = model.predict(test_x)

    # mae = np.mean(np.abs(test_y.values - pred_y))
    rmse = np.sqrt(np.mean(np.square(test_y.values - pred_y)))

    prediction_error = rmse

    # predictions for ml curve
    x = np.arange(0, df['wip'].max() + 0.1 , 0.01)
    y = model.predict(scalerx.transform(x.reshape(-1, 1)))

    # plot_curve(x, y, api, df)

    return training_loss, validation_loss, prediction_error


def evaluate_model(api, df):

    df = datasets.remove_outliers(df)

    infile = open('../../models/api_metrics/new_model/_scalars/scaler_' + api.replace('/', '_') + '.pkl', 'rb')
    scalerx = pkl.load(infile)
    infile.close()

    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    model = keras.models.load_model('../../models/api_metrics/new_model/' + api.replace('/', '_'), compile=False)
    
    # evaluation using bucket method
    error = []

    for i in range(5):
        test_sample = datasets.get_test_sample(test)
        test_x = scalerx.transform(test_sample['wip'].values.reshape(-1,1))
        test_y = test_sample['latency']

        pred_y = model.predict(test_x)

        # mae = np.mean(np.abs(test_y.values - pred_y))
        rmse = np.sqrt(np.mean(np.square(test_y.values - pred_y)))
        error.append(rmse)

    sample_error = np.mean(error)

    # evaluation with entire test set
    test_x = scalerx.transform(test['wip'].values.reshape(-1,1))
    test_y = test['latency'] 

    pred_y = model.predict(test_x)
    
    # mae = np.mean(np.abs(test_y.values - pred_y))
    rmse = np.sqrt(np.mean(np.square(test_y.values - pred_y)))
    prediction_error = rmse

    # predictions for ml curve
    x = np.arange(0, df['wip'].max() + 0.1 , 0.01)
    y = model.predict(scalerx.transform(x.reshape(-1, 1)))

    plot_curve(x, y, api, df)

    return prediction_error, sample_error


def plot_curve(x, y, api, df):
    # plt.yscale('log')
    plt.scatter(df['wip'], df['latency'], label='data')
    plt.plot(x, y, 'r', label='ml curve')
    plt.title(api)
    plt.xlabel('wip')
    plt.ylabel('latency')
    plt.legend()
    # plt.show()
    plt.savefig('../../Plots/new_plots/' + api.replace('/', '_') + '.png')
    plt.close()



def train_models():
    training_losses = []
    validation_losses = []
    prediction_errors = []

    for name, group in df:

        # if name == 'ballerina/http/Caller#respond':
        #     continue

        if name in api:
            continue
        
        print(name, '\n')

        training_loss, validation_loss, prediction_error = train_model(name, group)

        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        prediction_errors.append(prediction_error)

    mean_training_loss = np.mean(training_losses)
    mean_val_loss = np.mean(validation_losses)
    mean_prediction_error = np.mean(prediction_errors)

    median_training_loss = np.percentile(training_losses, 50)
    median_val_loss = np.percentile(validation_losses, 50)
    median_prediction_error = np.percentile(prediction_errors, 50)

    percentile95_training_loss = np.percentile(training_losses, 95)
    percentile95_val_loss = np.percentile(validation_losses, 95)
    percentile95_prediction_error = np.percentile(prediction_errors, 95)

    print('\n'.join([str(l) for l in training_losses]), '\n\n')
    print('\n'.join([str(l) for l in validation_losses]), '\n\n')
    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')

    print('Mean loss/val_loss/prediction_error', mean_training_loss, mean_val_loss, mean_prediction_error)
    print('Median loss/val_loss/prediction_error', median_training_loss, median_val_loss, median_prediction_error)
    print('95th percentile loss/val_loss/prediction_error', percentile95_training_loss, percentile95_val_loss, percentile95_prediction_error)


def evaluate_models():
    prediction_errors = []
    sample_errors =[]

    for name, group in df:

        if name == 'ballerina/http/Caller#respond':
            continue

        # if (name not in test_apis):
        #     continue

        prediction_error, sample_error = evaluate_model(name, group)

        prediction_errors.append(prediction_error)
        sample_errors.append(sample_error)

    mean_prediction_error = np.mean(prediction_errors)
    mean_sample_error = np.mean(sample_errors)

    median_prediction_error = np.percentile(prediction_errors, 50)
    median_sample_error = np.percentile(sample_errors, 50)

    percentile95_prediction_error = np.percentile(prediction_errors, 95)
    percentile95_sample_error = np.percentile(sample_errors, 95)

    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')
    print('\n'.join([str(l) for l in sample_errors]), '\n\n')

    print('Mean prediction_error/sample_error', mean_prediction_error, mean_sample_error)
    print('Median loss/prediction_error/sample_error', median_prediction_error, median_sample_error)
    print('95th percentile prediction_error/sample_error', percentile95_prediction_error, percentile95_sample_error)



def get_forecasts():
    for name, group in df:

        group = datasets.remove_outliers(group)

        infile = open('../../models/api_metrics/12_regression_epoch200_batch4_rmse_ouliers_removed_unscaledY/_scalars/scaler_' + name.replace('/', '_') + '.pkl', 'rb')
        scaler = pkl.load(infile)
        infile.close()

        model = keras.models.load_model('../../models/api_metrics/12_regression_epoch200_batch4_rmse_ouliers_removed_unscaledY/' + name.replace('/', '_'), compile=False)

        # predictions for ml curve
        x = np.arange(0, group['wip'].max() + 0.1 , 0.01)
        y = model.predict(scaler.transform(x.reshape(-1, 1)))
        ml_predictions[name] = y

    return ml_predictions

# train_models()
evaluate_models()