from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import datasets

EPSILON =  1e-6
top_5 = [
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#forward',
    'ballerina/http/HttpClient#post',
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
]
top_5_preds = {}

def fit_with_2deg_up_polynomial_regression_improved(x,y):
    coeffients = np.polyfit(x, y, 2)
    if coeffients[0] < 0:
        #this is not possible as wip increases latency must increase, so we try one degree polynomial 
        coeffients = np.polyfit(x, y, 1)
        coeffients = [0, coeffients[0], coeffients[1]]
        if coeffients[1] < 0:
            #this is not possible aslo, so we choose a constant function
            coeffients = [0, 0, np.mean(x)]
    print(coeffients[0], "x^2 +", coeffients[1], "x +", coeffients[2])
    polyregressor = np.poly1d(coeffients)
    # regresson_score = np.sqrt(mean_squared_error(y, polyregressor(x)))
    regresson_score = (np.sqrt(np.mean(np.square((y - polyregressor(x)) / (y + EPSILON))))) * 100
    print("polybomial regresson rmspe", regresson_score)    
    return PolyFit(coeffients, regresson_score)


class PolyFit:
    def __init__(self, coefficents, rmspe):
        self.coefficents = coefficents
        self.rmspe = rmspe


    def predict(self, x):
        x = np.array(x)
        return self.coefficents[0]* x**2 + self.coefficents[1]* x + self.coefficents[2]


def run():

    train_loss = []
    pred_loss = []
    # equation_params = []

    print("[INFO] loading data...")
    df = datasets.load_data()

    for name, group in df:

        print("\n", name)

        print("[INFO] constructing training/testing split...")
        (train, test) = train_test_split(group, test_size=0.3, random_state=42)

        trainX = train["wip"]
        testX = test["wip"]
        
        trainY = train["latency"]
        testY = test["latency"]

        print("[INFO] fittinig parameters...")
        model = fit_with_2deg_up_polynomial_regression_improved(trainX, trainY)
        print("Fitted Parameters: ", model.coefficents)

        train_loss.append(model.rmspe)

        # [a, b, c] = params
        # kappa = a /(a + b+  c)
        # sigma = (a + b) / (a + b + c)
        # lambd = 1 / (a + b + c)
        # equation_params.append([kappa, sigma, lambd])

        predY = model.predict(testX)
        # print("Preds : ", predY)

        #rmspe
        error = (np.sqrt(np.mean(np.square((testY - predY) / (testY + EPSILON))))) * 100
        
        # mean absolute percentage error
        # error = np.mean(np.abs((testY - predY) / (testY + EPSILON))) * 100
        
        print("Prediction Loss RMSPE: ", error)
        pred_loss.append(error)

        if name in top_5:
            preds = model.predict(np.arange(0, group.wip.max()))
            top_5_preds[name] = preds

    mean_train_loss = np.mean(train_loss)
    percentile_train_loss = np.percentile(train_loss, 95)
    
    mean_loss = np.mean(pred_loss)
    percentile_loss = np.percentile(pred_loss, 95)

    # print("\n".join([str(l) for l in equation_params]), "\n\n")
    print("\nTraining losses:")
    print("\n".join([str(l) for l in train_loss]), "\n\n")

    print("Mean train_loss", mean_train_loss)
    print("95th percentile train_loss", percentile_train_loss)

    print("\nPrediction losses:")
    print("\n".join([str(l) for l in pred_loss]), "\n\n")

    print("Mean loss", mean_loss)
    print("95th percentile loss", percentile_loss)

def domain_forecast():
    return top_5_preds

run()
