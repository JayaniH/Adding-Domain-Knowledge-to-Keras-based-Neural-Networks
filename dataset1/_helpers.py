import numpy as np

def print_errors(errors):

    mean_trainng_loss = np.mean(errors['training'])
    mean_val_loss = np.mean(errors['validation'])
    mean_prediction_error = np.mean(errors['prediction'])
    
    print('--------------------------------------------------------------------------------------------------------------------------------')
    print('Losses\n')
    print('Training Loss\n', '\n'.join([str(e) for e in errors['training']]), '\n')
    print('Mean training loss = ', mean_trainng_loss, '\n\n')
    print('Validation Loss\n', '\n'.join([str(e) for e in errors['validation']]), '\n')
    print('Mean validation loss = ', mean_val_loss, '\n\n')
    print('Prediction Error\n', '\n'.join([str(e) for e in errors['prediction']]), '\n')
    print('Mean prediction error = ', mean_prediction_error, '\n\n')

    return mean_trainng_loss, mean_val_loss, mean_prediction_error

def print_errors_eval(errors):

    mean_prediction_error = np.mean(errors['prediction'])
    mean_sample_error = np.mean(errors['sample'])
    
    print('--------------------------------------------------------------------------------------------------------------------------------')
    print('Errors\n')
    print('Prediction Error\n', '\n'.join([str(e) for e in errors['prediction']]), '\n')
    print('Mean prediction error = ', mean_prediction_error, '\n\n')
    print('Sample Error\n', '\n'.join([str(e) for e in errors['sample']]), '\n')
    print('Mean sample error = ', mean_sample_error, '\n\n')

    return  mean_prediction_error, mean_sample_error

def print_kfold(kfold):
    mean_k1 = np.mean(kfold['k1'])
    mean_k2 = np.mean(kfold['k2'])
    mean_k3 = np.mean(kfold['k3'])
    mean_k4 = np.mean(kfold['k4'])
    mean_k5 = np.mean(kfold['k5'])

    median_k1 = np.percentile(kfold['k1'], 50)
    median_k2 = np.percentile(kfold['k2'], 50)
    median_k3 = np.percentile(kfold['k3'], 50)
    median_k4 = np.percentile(kfold['k4'], 50)
    median_k5 = np.percentile(kfold['k5'], 50)

    percentile95_k1 = np.percentile(kfold['k1'], 95)
    percentile95_k2 = np.percentile(kfold['k2'], 95)
    percentile95_k3 = np.percentile(kfold['k3'], 95)
    percentile95_k4 = np.percentile(kfold['k4'], 95)
    percentile95_k5 = np.percentile(kfold['k5'], 95)

    print('K1\n', '\n'.join([str(k) for k in kfold['k1']]), '\n')
    print(mean_k1, '\n', median_k1, '\n', percentile95_k1, '\n')
    print('K2\n', '\n'.join([str(k) for k in kfold['k2']]), '\n')
    print(mean_k2, '\n', median_k2, '\n', percentile95_k2, '\n')
    print('K3\n', '\n'.join([str(k) for k in kfold['k3']]), '\n')
    print(mean_k3, '\n', median_k3, '\n', percentile95_k3, '\n')
    print('K4\n', '\n'.join([str(k) for k in kfold['k4']]), '\n')
    print(mean_k4, '\n', median_k4, '\n', percentile95_k4, '\n')
    print('K5\n', '\n'.join([str(k) for k in kfold['k5']]), '\n')
    print(mean_k5, '\n', median_k5, '\n', percentile95_k5, '\n')