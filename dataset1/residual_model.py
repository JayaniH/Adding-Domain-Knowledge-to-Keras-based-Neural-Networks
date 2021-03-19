import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datasets
import models
import domain_model3 as domain_model
import numpy as np
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
residual_models_predictions = {}

# load data
print('[INFO] loading data...')
df = datasets.load_data()

# fit domain model parameters
domain_model_parameters = domain_model.create_models()

def train_model(api, df):

    group = datasets.remove_outliers(df)
    
    df['domain_latency'] = domain_model.predict(api, df['wip'], domain_model_parameters[api])
    df['residuals'] = df['domain_latency'] - df['latency']

    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    train_y = train['residuals']
    test_y = test['residuals']

    scalerX = MinMaxScaler()
    train_x = scalerX.fit_transform(train['wip'].values.reshape(-1,1))
    test_x = scalerX.transform(test['wip'].values.reshape(-1,1))

    # save scaler X
    outfile = open('../../models/api_metrics/new_model/_scalars/scaler_' + api.replace('/', '_') + '.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_residual_model(train_x.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=200, batch_size=4)

    # save model
    model.save('../../models/api_metrics/new_model/' + api.replace('/', '_'))

    training_loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    pred_y = model.predict(test_x)

    # pred_y = residuals = domain_latency - latency
    # pred_latency = domain_latency  - (domain_latency - latency)

    # pred_latency = test['domain_latency'] - scalerY.inverse_transform(pred_y).flatten()
    pred_latency = test['domain_latency'] - pred_y.flatten()
    rmse = np.sqrt(np.mean(np.square(test['latency'].values - pred_latency)))
    # mae = np.mean(np.abs(test['latency'].values - pred_latency))

    prediction_error = rmse

    return training_loss, validation_loss, prediction_error

def evaluate_model(api, df):
    df = datasets.remove_outliers(df)

    df['domain_latency'] = domain_model.predict(api, df['wip'], domain_model_parameters[api])
    df['residuals'] = df['domain_latency'] - df['latency']

    infile = open('../../models/api_metrics/new_model/_scalars/scaler_' + api.replace('/', '_') + '.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    model = keras.models.load_model('../../models/api_metrics/new_model/' + api.replace('/', '_'), compile=False)

    # predictions for test set
    test_x = scalerX.transform(test['wip'].values.reshape(-1,1))
    test_y = test['residuals']
    pred_y = model.predict(test_x)

    # pred_latency = d_latency  - (d_latency - latency)
    pred_latency = test['domain_latency'] - pred_y.flatten()

    rmse = np.sqrt(np.mean(np.square(test['latency'].values - pred_latency)))
    # mae = np.mean(np.abs(test['latency'].values - pred_latency))
    prediction_error = rmse

    # evaluation using bucket method
    error = []
    pred_error = []

    for i in range(5):
        test_sample = datasets.get_test_sample(test)
        test_x = scalerX.transform(test_sample['wip'].values.reshape(-1,1))
        test_y = test_sample['residuals']

        # predict residual (domain_latency - latency)
        pred_y = model.predict(test_x)

        # residual error
        rmse = np.sqrt(np.mean(np.square(test_y.values - pred_y)))
        # mae = np.mean(np.abs(test_y.values - pred_y))
        error.append(rmse)

        # prediction error (latency)
        # pred_latency = test_sample['domain_latency'] - scalerY.inverse_transform(pred_y).flatten()
        pred_latency = test_sample['domain_latency'] - pred_y.flatten()
        rmse = np.sqrt(np.mean(np.square(test_sample['latency'].values - pred_latency)))
        # mae = np.mean(np.abs(test_sample['latency'].values - pred_latency))
        pred_error.append(rmse)

    sample_residual_error = np.mean(error)

    sample_prediction_error = np.mean(pred_error)

    # predictions for ml curve
    x = np.arange(0, df['wip'].max() + 0.1 , 0.01)
    domain_latency = domain_model.predict(api, x, domain_model_parameters[api])
    y = model.predict(scalerX.transform(x.reshape(-1, 1)))
    y = domain_latency - y.flatten()

    plot_curve(x, y, domain_latency, api, df)

    return prediction_error, sample_residual_error, sample_prediction_error


def plot_curve(x, y, domain_latency, api, df):

    # plt.yscale('log')
    plt.scatter(df['wip'], df['latency'], label='data')
    # plt.scatter(df['wip'], df['residuals'], label='data')
    plt.plot(x, y, 'r', label='residual model')
    plt.plot(x, domain_latency, 'g', label='domain model')
    plt.title(api)
    plt.xlabel('wip')
    plt.ylabel('latency')
    plt.legend()
    plt.show()
    # plt.savefig('../../Plots/residual_actual_domain/' + name.replace('/', '_') + '_loss.png')
    plt.close()


def train_models():
    training_losses = []
    validation_losses = []
    prediction_errors = []

    for name, group in df:

        if name not in test_apis:
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

    print('training loss\n')
    print('\n'.join([str(l) for l in training_losses]), '\n\n')
    print('validation loss\n')
    print('\n'.join([str(l) for l in validation_losses]), '\n\n')
    print('prediction error\n')
    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')

    print('Mean loss/val_loss/prediction_error/', mean_training_loss, mean_val_loss, mean_prediction_error)
    print('Median loss/val_loss/prediction_error/', median_training_loss, median_val_loss, median_prediction_error)
    print('95th percentile loss/val_loss/prediction_error/', percentile95_training_loss, percentile95_val_loss, percentile95_prediction_error)


def evaluate_models():
    prediction_errors = []
    sample_residual_errors = []
    sample_prediction_errors =[]
    for name, group in df:

        # if name == 'ballerina/http/Caller#respond':
        #     continue

        if (name not in test_apis):
            continue

        prediction_error, sample_residual_error, sample_prediction_error = evaluate_model(name, group)
        prediction_errors.append(prediction_error)
        sample_residual_errors.append(sample_residual_error)
        sample_prediction_errors.append(sample_prediction_error)
        
    mean_prediction_loss = np.mean(prediction_errors)
    mean_sample_loss = np.mean(sample_residual_errors)
    mean_sample_prediction_loss = np.mean(sample_prediction_errors)

    median_prediction_loss = np.percentile(prediction_errors, 50)
    median_sample_loss = np.percentile(sample_residual_errors, 50)
    median_sample_prediction_loss = np.percentile(sample_prediction_errors, 50)

    percentile95_prediction_loss = np.percentile(prediction_errors, 95)
    percentile95_sample_loss = np.percentile(sample_residual_errors, 95)
    percentile95_sample_prediction_loss = np.percentile(sample_prediction_errors, 95)

    print('residual error (bucket sampling)\n')
    print('\n'.join([str(l) for l in sample_residual_errors]), '\n\n')
    print('prediction error\n')
    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')
    print('prediction error (bucket sampling)\n')
    print('\n'.join([str(l) for l in sample_prediction_errors]), '\n\n')

    print('Mean sample_loss/prediction_loss/sample_predction_loss', mean_sample_loss, mean_prediction_loss, mean_sample_prediction_loss)
    print('Median sample_loss/prediction_loss/sample_prediction_loss', median_sample_loss, median_prediction_loss, median_sample_prediction_loss)
    print('95th percentile sample_loss/prediction_loss/sample_prediction_loss', percentile95_sample_loss, percentile95_prediction_loss, percentile95_sample_prediction_loss)

    return residual_models_predictions


def get_forecasts():

    for name, group in df:

        group = datasets.remove_outliers(group)

        group['domain_latency'] = domain_model.predict(name, group['wip'], domain_model_parameters[name])
        group['residuals'] = group['domain_latency'] - group['latency']

        infile = open('../../models/api_metrics/10_residual_rmse/_scalars/scalerX' + name.replace('/', '_') + '.pkl', 'rb')
        scalerX = pkl.load(infile)
        infile.close()

        model = keras.models.load_model('../../models/api_metrics/10_residual_rmse/' + name.replace('/', '_'), compile=False)

        x = np.arange(0, group['wip'].max() + 0.1 , 0.01)
        domain_latency = domain_model.predict(name, x, domain_model_parameters[name])
        y = model.predict(scalerX.transform(x.reshape(-1, 1)))
        y = domain_latency - y.flatten()
        residual_models_predictions[name] = y

        # print(name, y)

    return residual_models_predictions

# train_models()
# evaluate_models()