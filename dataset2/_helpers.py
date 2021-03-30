import numpy as np
import matplotlib.pyplot as plt  

def print_predictions(y_true, y_pred):
    print('\navg_response_time:\n','\n'.join([str(val) for val in y_true]))
    print('\npredicted avg_response_time:\n', '\n'.join([str(val) for val in y_pred]))

def plot_curve(df, x, y):
    plt.scatter(df['concurrent_users'], df['avg_response_time'], label='actual data')
    plt.plot(x, y, label='forecast')

    plt.title('Domain Model')
    plt.xlabel('concurrent_users')
    plt.ylabel('avg_response_time')
    plt.legend()
    # plt.show()
    # plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_msg_size.png')
    plt.close()

def get_average_error(error):
    print('\n--------------------------------')
    print('Prediction Errors\n')
    print('\n'.join([str(e) for e in error]), '\n\n')
    mean_error = np.mean(error)
    print('[RESULT] mean error = ', mean_error)
    return mean_error

def get_error(test, residual_prediction, response_time_prediction):
    rmse = np.sqrt(np.mean(np.square(test['avg_response_time'].values - response_time_prediction)))
    # mae = np.mean(np.abs(test['avg_response_time'].values - pred_response_time))
    domain_error = np.sqrt(np.mean(np.square(test['domain_prediction'] - test['avg_response_time'])))
    residual_error = np.sqrt(np.mean(np.square(test['residuals'] - residual_prediction)))

    rms_residuals = np.sqrt(np.mean(np.square(test['residuals'])))
    rms_avg_response_time = np.sqrt(np.mean(np.square(test['avg_response_time'])))

    print('[RESULT] prediction_error = ', rmse)
    print('[RESULT] domain_model_error = ', domain_error)
    print('[RESULT] residual_error = ', residual_error)
    print('[RESULT] percentage error residuals/ domain/ hybrid', (residual_error/rms_residuals) * 100 , (domain_error/rms_avg_response_time) * 100, (rmse/rms_avg_response_time) * 100)
    
    return rmse