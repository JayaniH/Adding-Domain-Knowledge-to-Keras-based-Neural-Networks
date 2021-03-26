import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import datasets
import models
import _helpers
import domain_model3 as domain_model
import numpy as np
import pickle as pkl


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



# load data
print('[INFO] loading data...')
df = datasets.load_data()

# fit domain model parameters
domain_model_parameters = domain_model.create_models()

def train_model(train_i, test_i, i, api, df):

    # df = datasets.remove_outliers(df)
    
    df['domain_latency'] = domain_model.predict(api, df['wip'], domain_model_parameters[api])
    df['residuals'] = df['domain_latency'] - df['latency']

    train = df.iloc[train_i]
    test = df.iloc[test_i]


    train_y = train['residuals']
    test_y = test['residuals']

    scalerX = MinMaxScaler()
    train_x = scalerX.fit_transform(train['wip'].values.reshape(-1,1))
    test_x = scalerX.transform(test['wip'].values.reshape(-1,1))

    # save scaler X
    outfile = open('../../models/api_metrics/new_model/_scalars/scalerX' + api.replace('/', '_') + str(i+1) +'.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_residual_model(train_x.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss='mean_absolute_error', optimizer=opt)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    print('[INFO] training model...')
    history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=200, batch_size=32 if train_x.shape[0] > 32 else 4)

    # save model
    model.save('../../models/api_metrics/new_model/' + api.replace('/', '_') + str(i+1))

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

def evaluate_model(train_i, test_i, i, api, df):
    # df = datasets.remove_outliers(df)

    df['domain_latency'] = domain_model.predict(api, df['wip'], domain_model_parameters[api])
    df['residuals'] = df['domain_latency'] - df['latency']

    infile = open('../../models/api_metrics/new_model/_scalars/scalerX' + api.replace('/', '_') + str(i+1) + '.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    train = df.iloc[train_i]
    test = df.iloc[test_i]

    model = keras.models.load_model('../../models/api_metrics/new_model/' + api.replace('/', '_') + str(i+1), compile=False)

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
    plt.plot(x, y, 'g', label='residual model')
    plt.plot(x, domain_latency, 'y', label='domain model')
    plt.title(api)
    plt.xlabel('wip')
    plt.ylabel('latency')
    plt.legend()
    # plt.show()
    plt.savefig('../../Plots/new_plots/' + api.replace('/', '_') + '_loss.png')
    plt.close()

def train_with_cross_validation(api, df):
    errors = {'training' : [], 'validation': [], 'prediction': []}

    df = datasets.remove_outliers(df)

    print('[INFO] constructing k fold split...')
    kf = KFold(n_splits=5, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('--------------------------------------------------------------------------------------------------------------------------------')
        print('K', str(i+1), '\n')
        training_loss, validation_loss, prediction_error = train_model(train_i, test_i, i, api, df)
        errors['training'].append(training_loss)
        errors['validation'].append(validation_loss)
        errors['prediction'].append(prediction_error)
        i += 1

    mean_trainng_loss, mean_val_loss, mean_prediction_error = _helpers.print_errors(errors)

    return mean_trainng_loss, mean_val_loss, mean_prediction_error


def evaluate_with_cross_validation(api, df):
    errors = {'prediction' : [], 'sample': []}

    df = datasets.remove_outliers(df)
    
    print('[INFO] constructing k fold split...')
    kf = KFold(n_splits=5, shuffle = True, random_state=14)
    i = 0

    for train_i, test_i in kf.split(df):
        print('--------------------------------------------------------------------------------------------------------------------------------')
        print('\nK', i+1)
        prediction_error, sample_residual_error, sample_prediction_error = evaluate_model(train_i, test_i, i, api, df)
        errors['prediction'].append(prediction_error)
        errors['sample'].append(sample_prediction_error)
        i += 1

    mean_prediction_error, mean_sample_error = _helpers.print_errors_eval(errors)

    return errors, mean_prediction_error, mean_sample_error


def train_models():
    training_losses = []
    validation_losses = []
    prediction_errors = []

    for name, group in df:

        print(name, '\n')
        training_loss, validation_loss, prediction_error = train_with_cross_validation(name, group)

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
    kfold = {'k1': [], 'k2': [], 'k3': [], 'k4': [], 'k5': []}
    prediction_errors = []
    sample_prediction_errors =[]

    for name, group in df:

        errors, prediction_error, sample_prediction_error = evaluate_with_cross_validation(name, group)
        [k1, k2, k3, k4, k5] = errors['prediction']
        kfold['k1'].append(k1)
        kfold['k2'].append(k2)
        kfold['k3'].append(k3)
        kfold['k4'].append(k4)
        kfold['k5'].append(k5)

        prediction_errors.append(prediction_error)
        sample_prediction_errors.append(sample_prediction_error)
        
    mean_prediction_loss = np.mean(prediction_errors)
    mean_sample_prediction_loss = np.mean(sample_prediction_errors)

    median_prediction_loss = np.percentile(prediction_errors, 50)
    median_sample_prediction_loss = np.percentile(sample_prediction_errors, 50)

    percentile95_prediction_loss = np.percentile(prediction_errors, 95)
    percentile95_sample_prediction_loss = np.percentile(sample_prediction_errors, 95)

    _helpers.print_kfold(kfold)
    
    print('prediction error\n')
    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')
    print('prediction error (bucket sampling)\n')
    print('\n'.join([str(l) for l in sample_prediction_errors]), '\n\n')

    print('Mean prediction_loss/sample_predction_loss', mean_prediction_loss, mean_sample_prediction_loss)
    print('Median prediction_loss/sample_prediction_loss', median_prediction_loss, median_sample_prediction_loss)
    print('95th percentile prediction_loss/sample_prediction_loss', percentile95_prediction_loss, percentile95_sample_prediction_loss)


# train_models()
evaluate_models()