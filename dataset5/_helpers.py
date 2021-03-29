import numpy as np
import matplotlib.pyplot as plt  

def get_error(y_true, y_pred):
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100 

    print('\n[RESULT] RMSE / MAE / MAPE .....', rmse, mae, mape, '\n')

    return rmse, mae, mape

def print_predictions(y_true, y_pred):
    print('\nTrue latency:\n','\n'.join([str(val) for val in y_true]))
    print('\nPredicted latency:\n', '\n'.join([str(val) for val in y_pred]))

def print_errors(errors):
    
    print('--------------------------------------------------------------------------------------------------------------------------------')
    print('Prediction Errors\n')
    print('RMSE\n', '\n'.join([str(e) for e in errors['rmse']]), '\n')
    print('Mean RMSE = ', np.mean(errors['rmse']), '\n\n')
    print('MAE\n', '\n'.join([str(e) for e in errors['mae']]), '\n')
    print('Mean MAE = ', np.mean(errors['mae']), '\n\n')
    print('MAPE\n', '\n'.join([str(e) for e in errors['mape']]), '\n')
    print('Mean MAPE = ', np.mean(errors['mape']), '\n\n')

def plot_curve(df, x, y):
    plt.scatter(df['concurrent_users'], df['latency'], label='actual data')
    plt.plot(x, y, label='forecast')

    plt.title('[curve_fit]\nDomain Model')
    plt.xlabel('concurrent_users')
    plt.ylabel('mid_latency')
    plt.legend()
    plt.show()
    # plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_cores.png')
    plt.close()

def plot_mape(y_true, y_pred):
    ape = np.abs((y_true - y_pred)/y_true)*100 
    plt.scatter(y_true, ape, label='absolute percentage error')

    plt.title('MAPE')
    plt.xlabel('actual_latency')
    plt.ylabel('absolute_percentage_error')
    plt.legend()
    plt.show()
    # plt.savefig('../../Plots/_api_manager/18_domain_model_minimization_eq2_regularization_param_10000a_100b_10a1/' + str(i+1) + '_cores.png')
    plt.close()

def print_parameters_and_errors(param_estimates, errors):
    print('----------------------')
    print("Parameter Estimates\n")
    print("Sigma(s)\n", "\n".join([str(s) for s in param_estimates['s']]), "\n\n")
    print("Kappa(k)\n", "\n".join([str(k) for k in param_estimates['k']]), "\n\n")
    print("Lambda(l)\n", "\n".join([str(l) for l in param_estimates['l']]), "\n\n")
    
    print('----------------------')
    print('Prediction Errors\n')
    print('RMSE\n', '\n'.join([str(e) for e in errors['rmse']]), '\n')
    print('Mean RMSE = ', np.mean(errors['rmse']), '\n\n')
    print('MAE\n', '\n'.join([str(e) for e in errors['mae']]), '\n')
    print('Mean MAE = ', np.mean(errors['mae']), '\n\n')
    print('MAPE\n', '\n'.join([str(e) for e in errors['mape']]), '\n')
    print('Mean MAPE = ', np.mean(errors['mape']), '\n\n')