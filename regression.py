import matplotlib.pyplot as plt 
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import datasets
import models
import numpy as np
import pandas as pd


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 100

loss = []
validation_loss = []
top_5 = [
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#forward',
    'ballerina/http/HttpClient#post',
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
]
test_apis = [
    'ballerina/http/Client#forward#http://13.90.39.240:8602',
    'ballerina/http/Client#get#https://ap15.salesforce.com',
    'ballerina/http/Client#patch#https://graph.microsoft.com',
    'ballerina/http/Client#post#https://login.salesforce.com/services/oauth2/token',
    'ballerinax/sfdc/SObjectClient#createRecord'
]
top_5_preds = {}
test_preds_regression = {}

results_file = open("./results/regression_results.txt", "w")

# load data
print("[INFO] loading data...")
df = datasets.load_data()

for name, group in df:

    # if name == "ballerina/http/Caller#respond":
    #     continue

    if (name not in top_5) and (name not in test_apis):
        continue

    print(name, "\n")
    
    results_file.write(name)
    results_file.write("\n")

    print("[INFO] constructing training/testing split...")
    (train, test) = train_test_split(group, test_size=0.3, random_state=42)

    #scale Y
    maxLatency = train["latency"].max()
    trainY = train["latency"] / maxLatency
    testY = test["latency"] / maxLatency

    print("[INFO] processing data...")
    (trainX, testX) = datasets.process_attributes(train, test)

    model = models.create_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-2, decay=1e-3/200)
    model.compile(loss=root_mean_squared_percentage_error, optimizer=opt)

    print("[INFO] training model...")
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=8)
    
    # get final loss
    loss.append(history.history['loss'][-1])
    validation_loss.append(history.history['val_loss'][-1])

    # record results
    print(history.history)


    # results_file.write(str(history.history))
    # results_file.write("\n\n")

    # model.evaluate(testX, testY, batch_size=8)

    # plot results
    # plt.plot(list(range(1,201)), history.history['loss'], label="loss")
    # plt.title(name)
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.ylim(0, 1)

    # plt.plot(list(range(1,201)), history.history['val_loss'], label="val_loss")
  
    # plt.legend()

    # plt.show()
    # plt.savefig('./Plots/loss/' + name.replace("/", "_") + '_loss.png')
    # plt.close()

    x = np.arange(0, group.wip.max(), 0.1)
    # preds = model.predict(x)
    # test_preds_regression[name] = preds

    # results = pd.DataFrame({'x': x, 'y_pred': preds})
    # results_file.write(str(results))
    # results_file.write(str(preds))


    if name in top_5:
        preds = model.predict(x)
        top_5_preds[name] = preds
        # print('preds', name, preds)

    if name in test_apis:
        preds = model.predict(x)
        test_preds_regression[name] = preds
        # print('preds', name, preds)



print(test_preds_regression)

mean_loss = np.mean(loss)
mean_val_loss = np.mean(validation_loss)

percentile_loss = np.percentile(loss, 95)
percentile_val_loss = np.percentile(validation_loss, 95)

print("Mean loss/val_loss", mean_loss, mean_val_loss)
print("95th percentile loss/val_loss", percentile_loss, percentile_val_loss)

# results_file.write("\nMean loss/val_loss %s / %s" % (mean_loss, mean_val_loss)) 
# results_file.write("\n95th percentile loss/val_loss %s / %s" % (percentile_loss, percentile_val_loss)) 

results_file.write("\ntop 5")
results_file.write(str(top_5_preds))
results_file.write("\ntest")
results_file.write(str(test_preds_regression))
results_file.close()

def regression_forecast():
    return top_5_preds, test_preds_regression