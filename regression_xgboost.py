from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd 
import numpy as np
import datasets

loss = []
EPSILON =  1e-6

high_loss_apis = [
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/Client#get#http://postman-echo.com',
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
    'ballerina/http/Client#post#http://airline-mock-svc:8090',
    'ballerina/http/Client#post#http://car-mock-svc:8090',
    'ballerina/http/HttpClient#forward',
    # 'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#post',
    'ballerina/http/HttpClient#get',
    'ballerina/http/HttpCachingClient#post'
]
test_apis = [
    'ballerina/http/Client#get#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://ap15.salesforce.com',
    'ballerina/http/Client#post#https://login.salesforce.com/services/oauth2/token',
    'ballerinax/sfdc/QueryClient#getQueryResult',
    'ballerinax/sfdc/SObjectClient#createOpportunity'
]
top_5_preds_xgb = {}
test_preds_xgb = {}

df = datasets.load_data()

for name, group in df:
    if (name not in high_loss_apis) and (name not in test_apis):
        continue

    X = group["wip"].values.reshape(group["wip"].shape[0],1)
    y = group["latency"]

    data_dmatrix = xgb.DMatrix(data=X, label=y)

    # print(X.shape,y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

    xg_reg.fit(X_train, y_train)

    preds = xg_reg.predict(X_test)

    # rmspe = (np.sqrt(np.mean(np.square((y_test - preds) / (y_test + EPSILON))))) * 100
    mae = np.mean(np.abs(y_test - preds))
    loss.append(mae)

    preds = xg_reg.predict(np.arange(0, group.wip.max() + 0.1, 0.01).reshape(-1, 1))
    test_preds_xgb[name] = preds

    # print(name, preds)

# print(top_5_preds_xgb)

mean_loss = np.mean(loss)

percentile_loss = np.percentile(loss, 95)

print("-------Regression XGBoost--------")
print("\n".join([str(l) for l in loss]), "\n\n")

print("Mean loss", mean_loss)
print("95th percentile loss", percentile_loss)

def xgb_regression_forecast():
    return test_preds_xgb