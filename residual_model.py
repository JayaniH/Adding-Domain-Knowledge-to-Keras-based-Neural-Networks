import matplotlib.pyplot as plt 
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import datasets
import models
import domain_model
import numpy as np
import argparse
import locale
import os


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_percentage_error(y_true, y_pred):
    EPSILON =  1e-6
    return (K.sqrt(K.mean(K.square((y_true - y_pred) / (y_true + EPSILON))))) * 100

loss = []
validation_loss = []
EPSILON =  1e-6

# results_file = open("./results/residual_results.txt", "w")

# load data
print("[INFO] loading data...")
df = datasets.load_data()

# run domain model
domain_model.run()

for name, group in df:
    print(name, "\n")

    # results_file.write(name)
    # results_file.write("\n")
    
    group["domainY"] = domain_model.predict(name, group["wip"])

    print("[INFO] constructing training/testing split...")
    (train, test) = train_test_split(group, test_size=0.3, random_state=42)

    #scale Y
    maxY = (train["domainY"] - train["latency"]).max()
    trainY = (train["domainY"] - train["latency"])
    testY = (test["domainY"] - test["latency"])

    print("[INFO] processing data...")
    (trainX, testX) = datasets.process_attributes(train, test)

    model = models.create_residual_model(trainX.shape[1])
    opt = Adam(learning_rate=1e-3, decay=1e-3/200)
    model.compile(loss=root_mean_squared_percentage_error, optimizer=opt)

    print("[INFO] training model...")
    history = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=200, batch_size=8)

    print("[INFO] predicting latency...")
    preds = model.predict(testX)

    # predY = d_latency  - (d_latency - latency)
    predY = test["domainY"]- preds.flatten()

    # prediction error
    rmspe = (np.sqrt(np.mean(np.square((test["latency"] - predY) / (test["latency"] + EPSILON))))) * 100

    # get final loss
    loss.append(rmspe)

    # record results
    # results_file.write(str(testY))
    # results_file.write("\n----------\n")
    # results_file.write(str(predY))
    # results_file.write("\n\n")


mean_loss = np.mean(loss)

percentile_loss = np.percentile(loss, 95)

print("Mean loss", mean_loss)
print("95th percentile loss", percentile_loss)

print("\n".join([str(l) for l in loss]), "\n\n")

# results_file.write(str(loss)) 

# results_file.write("\nMean loss %s " % (mean_loss)) 
# results_file.write("\n95th percentile loss %s " % (percentile_loss)) 

# results_file.close()