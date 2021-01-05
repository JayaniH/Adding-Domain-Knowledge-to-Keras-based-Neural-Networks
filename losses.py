from tensorflow import keras
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 10

def custom_loss(y, y_pred):
    y_true = y[:,0]
    domain_latency = y[:,1]

    # print("custom loss")
    # y_true=K.print_tensor(y_true)
    # domain_latency=K.print_tensor(domain_latency)

    return K.sqrt(K.mean(K.square(y_pred - y_true))) +  0.1 * K.sqrt(K.mean(K.square(domain_latency - y_pred)))

def custom_loss_clipping(y, y_pred):
    y_true = y[:,0]
    domain_latency = y[:,1]

    threshold = 0.5 * K.mean(y_true)

    loss = K.sqrt(K.mean(K.square(y_pred - y_true)))

    return loss if loss <= threshold else (loss + 0.2 * K.sqrt(K.mean(K.square(domain_latency - y_pred))))

def custom_loss_dynamic_threshold(threshold):

    def loss(y, y_pred):
        y_true = y[:,0]
        domain_latency = y[:,1]
    
        loss = K.sqrt(K.mean(K.square(y_pred - y_true)))

        return loss if loss <= threshold else (loss + 0.2 * K.sqrt(K.mean(K.square(domain_latency - y_pred))))

    return loss



def custom_loss_approximation(y_l, y_u):

    def loss(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) + K.relu(y_l - y_pred) + K.relu(y_pred - y_u)

    return loss