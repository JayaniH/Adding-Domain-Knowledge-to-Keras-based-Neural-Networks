from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd 
import numpy as np
import datasets

loss = []
EPSILON =  1e-6

top_5 = [
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#forward',
    'ballerina/http/HttpClient#post',
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
]
top_5_preds = {}

df = datasets.load_data()

for name, group in df:
    X = group["wip"].values.reshape(group["wip"].shape[0],1)
    y = group["latency"]

    data_dmatrix = xgb.DMatrix(data=X, label=y)

    # print(X.shape,y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

    xg_reg.fit(X_train, y_train)

    preds = xg_reg.predict(X_test)

    rmspe = (np.sqrt(np.mean(np.square((y_test - preds) / (y_test + EPSILON))))) * 100
    loss.append(rmspe)

    # forecast for top 5 APIs
    if name in top_5:
        preds = xg_reg.predict(np.arange(0, int(group.wip.max()) + 1).reshape(int(group.wip.max())+ 1,1))
        top_5_preds[name] = preds

print(top_5_preds)

mean_loss = np.mean(loss)

percentile_loss = np.percentile(loss, 95)

print("\n".join([str(l) for l in loss]), "\n\n")

print("Mean loss", mean_loss)
print("95th percentile loss", percentile_loss)

def xgb_regression_forecast():
    return top_5_preds