import numpy as np
import pandas as pd
import regression
import regression_xgboost
import domain_model2
import seaborn as sns
import matplotlib.pyplot as plt 

top_5 = [
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#forward',
    'ballerina/http/HttpClient#post',
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
]

df = pd.read_csv('performance_data_truncated.csv', sep="\t")
df = df[df['api_name'].isin(top_5)]
df = df[df['wip'] < 1500]

xgboost_preds = regression_xgboost.xgb_regression_forecast()

regression_preds = regression.regression_forecast()

domain_model2.run()
domain_preds = domain_model2.domain_forecast()

# data_plot = pd.DataFrame({"api_name":top_5, "wip": df.wip, "latency": df.latency, "xgboost": xgboost_preds, "domain": domain_preds, "wip_all": np.arange(0, 1500)})
# g = sns.FacetGrid(data_plot, col="api_name", col_wrap=5)
# g.map(sns.scatterplot, "wip", "latency", edgecolor="w").add_legend()
# # g.map(sns.lineplot, "wip_all", "regression", edgecolor="w").add_legend()
# g.map(sns.lineplot, "wip_all", "xgboost", edgecolor="w").add_legend()
# g.map(sns.lineplot, "wip_all", "domain", edgecolor="w").add_legend()
# plt.show()

print("domain: ", domain_preds)
print("xgboost:" ,xgboost_preds)
print("regression: ", regression_preds)

df = df.groupby(by="api_name")

for name, group in df:
    plt.scatter(group.wip, group.latency)
    print(np.arange(0, group.wip.max()).shape, xgboost_preds[name].shape, domain_preds[name].shape)
    plt.plot(np.arange(0, group.wip.max()), regression_preds[name], 'r', label='regression')
    plt.plot(np.arange(0, group.wip.max()), xgboost_preds[name], 'm', label='xgboost')
    plt.plot(np.arange(0, group.wip.max()), domain_preds[name], 'y', label='domain')
    plt.title(name)
    plt.xlabel('wip')
    plt.ylabel('latency')
    plt.legend()
    plt.show()