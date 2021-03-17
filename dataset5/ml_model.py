import matplotlib.pyplot as plt 
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import models
import _helpers
import numpy as np
import pandas as pd
import pickle as pkl
import random

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

# load data
print('[INFO] loading data...')
df = pd.read_csv('springboot_summary_truncated.csv', sep=',')

def train_model():

    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    print('[INFO] processing data...')

    trainY = train['latency']
    testY = test['latency']

    scalerX = MinMaxScaler()
    trainX = scalerX.fit_transform(train[['concurrent_users', 'heap_size', 'collector', 'size',  'use_case']].values.reshape(-1,5))
    testX = scalerX.transform(test[['concurrent_users', 'heap_size', 'collector', 'size',  'use_case']].values.reshape(-1,5))

    # save scaler X
    outfile = open('../../models/springboot2/new_model/_scalars/scalerX.pkl', 'wb')
    pkl.dump(scalerX, outfile)
    outfile.close()

    model = models.create_model(trainX.shape[1])
    opt = Adam(learning_rate=0.01)
    model.compile(loss=root_mean_squared_error, optimizer=opt)

    print('[INFO] training model...')
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=4)

    # save model
    model.save('../../models/springboot2/new_model/model')

    loss = history.history['loss'][-1]
    validation_loss = history.history['val_loss'][-1]

    print('[INFO] predicting latency...')
    latency_prediction = model.predict(testX)
    
    print('[RESULT] loss / val_loss', loss, validation_loss)
    rmse, mae, mape = _helpers.get_error(testY.values, latency_prediction.flatten())   

    return rmse, mae, mape


def evaluate_model():

    infile = open('../../models/springboot2/new_model/_scalars/scalerX.pkl', 'rb')
    scalerX = pkl.load(infile)
    infile.close()

    print('[INFO] constructing training/testing split...')
    (train, test) = train_test_split(df, test_size=0.3, random_state=42)

    model = keras.models.load_model('../../models/springboot2/new_model/model', compile=False)

    # preds for dataset
    testX = scalerX.transform(test[['concurrent_users', 'heap_size', 'collector', 'size',  'use_case']].values.reshape(-1,5))
    testY = test['latency']

    latency_prediction = model.predict(testX)

    _helpers.print_predictions(testY.values, latency_prediction.flatten())

    results_df = pd.DataFrame({'concurrent_users': test['concurrent_users'], 'heap_size': test['heap_size'], 'collector': test['collector'], 'latency': testY, 'prediction': latency_prediction.flatten()})
    print(results_df)
    results_df.to_csv('../../models/springboot2/new_model/results/result.csv', sep=",", index= False)

    rmse, mae, mape = _helpers.get_error(testY.values, latency_prediction.flatten())
    
    return rmse, mae, mape


# train_model()
evaluate_model()