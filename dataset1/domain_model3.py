
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import datasets

# domain model using curve fit


# USL equation
def USL(n, s, k, l):
	return (1 + s*(n-1) + k*n*(n-1))/l


print('[INFO] loading data...')
df = datasets.load_data()


def create_model(api, df):

    df = datasets.remove_outliers(df)

    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    train_x = train['wip']
    train_y = train['latency']

    params, cov = curve_fit(USL, train_x, train_y, p0=[0.5, 0.5, 0.5], bounds=([0,0,0],[1, 1, np.inf]))

    print('Fitteed Parameters: ', params)
    [s, k, l] = params

    # evaluation using bucket method
    error = []

    for i in range(5):
        test_sample = datasets.get_test_sample(test)
        test_x = test_sample['wip']
        test_y = test_sample['latency']

        pred_y = USL(test_x, s, k, l)
        # mae = np.mean(np.abs(test_y - pred_y))
        rmse = np.sqrt(np.mean(np.square(test_y - pred_y)))
        error.append(rmse)

        # print('test sample ', i, ' ', test.shape, test_sample.shape, test_y.mean(), mae)

    sample_error = np.mean(error)

    # evaluation without bucket sampling (using the whole dataset)
    test_x = test['wip']
    test_y = test['latency']

    pred_y = USL(test_x, s, k, l)

    # mae = np.mean(np.abs(test_y - pred_y))
    rmse = np.sqrt(np.mean(np.square(test_y - pred_y)))
    prediction_error = rmse

    x = np.arange(0, df['wip'].max() +0.1 , 0.01)
    y = USL(x, s, k, l)

    return params, prediction_error, sample_error


def plot_curve(x, y, api, df):
    # plt.yscale('log')
    plt.scatter(df['wip'], df['latency'])
    plt.plot(x, y, 'y', label='domain')
    plt.title('[curve_fit]\n'+api)
    plt.xlabel('wip')
    plt.ylabel('latency')
    plt.legend()
    # plt.show()
    # plt.savefig('../../Plots/new_plots/' + name.replace('/', '_') + '.png')
    plt.close()


def create_models():

    parameters = {}
    sample_errors = []
    prediction_errors = []
    equation_params = []
    estimate_s = []
    estimate_k = []
    estimate_l = []

    for name, group in df:

        print('\n', name)

        params, prediction_error, sample_error = create_model(name, group)

        parameters[name] = params
        prediction_errors.append(prediction_error)
        sample_errors.append(sample_error)

        equation_params.append(params)
        [s, k, l] = params
        estimate_s.append(s)
        estimate_k.append(k)
        estimate_l.append(l)

 
    mean_prediction_error = np.mean(prediction_errors)
    median_prediction_error = np.percentile(prediction_errors, 50)
    percentile95_prediction_error = np.percentile(prediction_errors, 95)

    mean_sample_error = np.mean(sample_error)
    median_sample_error = np.percentile(sample_error, 50)
    percentile95_sample_error = np.percentile(sample_error, 95)

    print('-------Domain Model Results--------\n')
    print('Parameter Estimates\n')
    print('\n'.join([str(l) for l in equation_params]), '\n\n')

    print('Parameter Estimates - Sigma(s)\n')
    print('\n'.join([str(s) for s in estimate_s]), '\n\n')
    print('Parameter Estimates - Kappa(k)\n')
    print('\n'.join([str(k) for k in estimate_k]), '\n\n')
    print('Parameter Estimates - Lambda(l)\n')
    print('\n'.join([str(l) for l in estimate_l]), '\n\n')

    print('Prediction Error\n')
    print('\n'.join([str(l) for l in prediction_errors]), '\n\n')
    print('Prediction Error (bucket sampling)\n')
    print('\n'.join([str(l) for l in sample_errors]), '\n\n')

    print('Mean prediction_error/sample_error', mean_prediction_error, mean_sample_error)
    print('Median prediction_error/sample_error', median_prediction_error, median_sample_error)
    print('95th percentile prediction_error/sample_error', percentile95_prediction_error, percentile95_sample_error)

    return parameters


def predict(api, wip, parameters):
    print('[INFO] predicting domain latency for ' + api)

    # [s, k, l] = parameters[api]
    [s, k, l] = parameters
    d_latency = USL(wip, s, k, l)

    return d_latency


def get_forecasts():

    domain_model_predictions ={}

    parameters = create_models()
    for name, group in df:
        group = datasets.remove_outliers(group)
        x = np.arange(0, group['wip'].max() +0.1 , 0.01)
        y = predict(name, x, parameters[name])
        domain_model_predictions[name] = y

    return domain_model_predictions

# create_models()