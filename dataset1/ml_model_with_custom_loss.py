import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datasets
import models
import losses
import domain_model3 as domain_model
import numpy as np
import pickle as pkl

# load data
print('[INFO] loading data...')
df = datasets.load_data()

# fit domain model parameters
domain_model_parameters = domain_model.create_models()

def train_model(api, df):
    df = datasets.remove_outliers(df)

    df['domain_latency'] = domain_model.predict(api, df['wip'], domain_model_parameters[api])

    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    print('[INFO] processing data...')

    #scaling
    train_y = train[['latency', 'domain_latency']]
    test_y = test[['latency', 'domain_latency']]

    y_lower = np.mean(train_y['latency']) - 3 * np.std(train_y['latency'])
    y_upper = np.mean(train_y['latency']) + 3 * np.std(train_y['latency'])

    clipping_threshold = 0.1 * np.mean(train_y['latency'])

    scalerX = MinMaxScaler()
    train_x = scalerX.fit_transform(train['wip'].values.reshape(-1,1))
    test_x = scalerX.transform(test['wip'].values.reshape(-1,1))

    # save scaler X
    outfile = open('../../models/api_metrics/new_model/_scalars/scalerX' + api.replace('/', '_') + '.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_model(train_x.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=losses.custom_loss_approximation_domain_regularization_clipped(y_lower, y_upper, clipping_threshold), optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=train_x, y=train_y, validation_data=(test_x, test_y), epochs=200, batch_size=4)

    # save model
    model.save('../../models/api_metrics/new_model/' + api.replace('/', '_'))

    training_loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    pred_y = model.predict(test_x)

    rmse = np.sqrt(np.mean(np.square(test['latency'].values - pred_y)))
    # mae = np.mean(np.abs(test['latency'].values - pred_y))
    
    prediction_error = rmse

    return training_loss, validation_loss, prediction_error

def evaluate_model(api, df):
    df = datasets.remove_outliers(df)

    df['domain_latency'] = domain_model.predict(api, df['wip'], domain_model_parameters[api])

    infile = open('../../models/api_metrics/12_regression_approximation_mean_3std_1_regularization_01/_scalars/scalerX' + api.replace('/', '_') + '.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    model = keras.models.load_model('../../models/api_metrics/12_regression_approximation_mean_3std_1_regularization_01/' + api.replace('/', '_'), compile=False)

    # evaluation using bucket method
    pred_error = []

    for i in range(5):
        test_sample = datasets.get_test_sample(test)
        test_x = scalerX.transform(test_sample['wip'].values.reshape(-1,1))

        # predict residual (domain_latency - latency)
        pred_y = model.predict(test_x)

        # prediction error (latency)
        rmse = np.sqrt(np.mean(np.square(test_sample['latency'].values - pred_y)))
        # mae = np.mean(np.abs(test_sample['latency'].values - pred_y))
        pred_error.append(rmse)

        # print('test sample ', i, ' ', test.shape, test_sample.shape, test_y.mean(), mae)

    sample_error = np.mean(pred_error)

    # predictions for test set
    test_x = scalerX.transform(test['wip'].values.reshape(-1,1))
    pred_y = model.predict(test_x)

    rmse = np.sqrt(np.mean(np.square(test['latency'].values - pred_y)))
    # mae = np.mean(np.abs(test['latency'].values - pred_y))
    prediction_error = rmse

    # predictions for ml curve
    x = np.arange(0, df['wip'].max() + 0.1 , 0.01)
    y = model.predict(scalerX.transform(x.reshape(-1, 1)))

    # plot_curve(x, y, api, df)

    return prediction_error, sample_error

def plot_curve(x, y, api, df):
    # plt.yscale('log')
    plt.scatter(df['wip'], df['latency'], label='data')
    plt.plot(x, y, 'r', label='ml')
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

    percentile_training_loss = np.percentile(training_losses, 95)
    percentile_val_loss = np.percentile(validation_losses, 95)
    percentile_prediction_error = np.percentile(prediction_errors, 95)

    print('\n'.join([str(l) for l in training_losses]), '\n\n')
    print('\n'.join([str(l) for l in validation_losses]), '\n\n')
    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')

    print('Mean training_loss/val_loss/prediction_error', mean_training_loss, mean_val_loss, mean_prediction_error)
    print('Median training_loss/val_loss/prediction_error', median_training_loss, median_val_loss, median_prediction_error)
    print('95th percentile training_loss/val_loss/prediction_error', percentile_training_loss, percentile_val_loss, percentile_prediction_error)


def evaluate_models():
    prediction_errors =[]
    sample_errors = []

    for name, group in df:
            
        prediction_error, sample_error = evaluate_model(name, group)
        prediction_errors.append(prediction_error)
        sample_errors.append(sample_error)

    mean_prediction_error = np.mean(prediction_errors)
    mean_sample_error = np.mean(sample_errors)

    median_prediction_error = np.percentile(prediction_errors, 50)
    median_sample_error = np.percentile(sample_errors, 50)

    percentile_prediction_error = np.percentile(prediction_errors, 95)
    percentile_sample_error = np.percentile(sample_errors, 95)

    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')
    print('\n'.join([str(l) for l in sample_errors]), '\n\n')

    print('Mean prediction_error/sample_error', mean_prediction_error, mean_sample_error)
    print('Median prediction_error/sample_error', median_prediction_error, median_sample_error)
    print('95th percentile prediction_error/sample_error', percentile_prediction_error, percentile_sample_error)


def get_forecasts():
    ml_with_custom_loss_predictions = {}

    for name, group in df:

        group = datasets.remove_outliers(group)

        group["domain_latency"] = domain_model.predict(name, group["wip"], domain_model_parameters[name])

        infile = open("../../models/api_metrics/12_regression_approximation_mean_3std_1_regularization_01/_scalars/scalerX" + name.replace("/", "_") + ".pkl", "rb")
        scalerX = pkl.load(infile)
        infile.close()

        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        model = keras.models.load_model('../../models/api_metrics/12_regression_approximation_mean_3std_1_regularization_01/' + name.replace("/", "_"), compile=False)

        # preds for ml curve
        x = np.arange(0, group["wip"].max() + 0.1 , 0.01)
        y = model.predict(scalerX.transform(x.reshape(-1, 1)))
        ml_with_custom_loss_predictions[name] = y

    return ml_with_custom_loss_predictions


# train_models()
evaluate_models() 