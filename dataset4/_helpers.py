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