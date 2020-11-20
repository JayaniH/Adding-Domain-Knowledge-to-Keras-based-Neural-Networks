import numpy as np
import pandas as pd
import regression
import regression_xgboost
import domain_model2
import seaborn as sns
import matplotlib.pyplot as plt 

high_loss_apis = [
    'ballerina/http/Client#get#https://covidapi.info/api/v1',
    'ballerina/http/Client#get#http://postman-echo.com',
    'ballerina/http/Client#post#http://airline-mock-svc:8090',
    'ballerina/http/Client#post#http://car-mock-svc:8090',
    'ballerina/http/Client#post#http://hotel-mock-svc:8090',
    'ballerina/http/HttpCachingClient#forward',
    'ballerina/http/Caller#respond',
    'ballerina/http/HttpClient#get',
    'ballerina/http/HttpClient#post',
    'ballerina/http/HttpCachingClient#post'
]
test_apis = [
    'ballerina/http/Client#forward#http://13.90.39.240:8602',
    'ballerina/http/Client#get#https://ap15.salesforce.com',
    'ballerina/http/Client#patch#https://graph.microsoft.com',
    'ballerina/http/Client#post#https://login.salesforce.com/services/oauth2/token',
    'ballerinax/sfdc/SObjectClient#createRecord'
]

df = pd.read_csv('performance_data_truncated.csv', sep="\t")
df = df[df['wip'] < 1500]

df1 = df[df['api_name'].isin(high_loss_apis)]
df2 = df[df['api_name'].isin(test_apis)]

regression.run_regression()
domain_model2.run()

xgboost_top5_preds, xgboost_test_preds = regression_xgboost.xgb_regression_forecast()

regression_top5__preds, regression_test_preds = regression.regression_forecast()

domain_top5_preds, domain_test_preds = domain_model2.domain_forecast()

# data_plot = pd.DataFrame({"api_name":high_loss_apis, "wip": df.wip, "latency": df.latency, "xgboost": xgboost_preds, "domain": domain_preds, "wip_all": np.arange(0, 1500)})
# g = sns.FacetGrid(data_plot, col="api_name", col_wrap=5)
# g.map(sns.scatterplot, "wip", "latency", edgecolor="w").add_legend()
# # g.map(sns.lineplot, "wip_all", "regression", edgecolor="w").add_legend()
# g.map(sns.lineplot, "wip_all", "xgboost", edgecolor="w").add_legend()
# g.map(sns.lineplot, "wip_all", "domain", edgecolor="w").add_legend()
# plt.show()

# print("domain: ", domain_preds)
# print("xgboost:" ,xgboost_preds)
# print("regression: ", regression_preds)

df1 = df1.groupby(by="api_name")
df2 = df2.groupby(by="api_name")

results_file = open("./results/predictions.txt", "w")

# print("Top 5 \n")
# results_file.write("\n\n Top 10 APIs predictions")
# for name, group in df1:

#     results_file.write(name)
#     results_file.write("\n")
#     results_file.write(str(regression_test_preds[name]))
#     results_file.write("\n")

#     x = np.arange(0, group.wip.max() +1 , 0.1)

#     plt.scatter(group.wip, group.latency)
#     plt.plot(x, regression_test_preds[name], 'r', label='regression')
#     plt.plot(np.arange(0, int(group.wip.max()) + 1, 0.1), xgboost_test_preds[name], 'm', label='xgboost')
#     plt.plot(x, domain_test_preds[name], 'y', label='domain')
#     plt.yscale("log")
#     plt.title(name)
#     plt.xlabel('wip')
#     plt.ylabel('latency')
#     plt.legend()
#     # plt.show()
#     plt.savefig('../Plots/forecasts/' + name.replace("/", "_") + '_highloss.png')
#     plt.close()

# print("Test \n")
# results_file.write("\n\n Test APIs predictions")
# for name, group in df2:

#     results_file.write(name)
#     results_file.write("\n")
#     results_file.write(str(regression_test_preds[name]))
#     results_file.write("\n")

#     x = np.arange(0, group.wip.max() +1 , 0.1)

#     plt.scatter(group.wip, group.latency)
#     plt.plot(x, regression_test_preds[name], 'r', label='regression')
#     plt.plot(np.arange(0, int(group.wip.max()) + 1, 0.1), xgboost_test_preds[name], 'm', label='xgboost')
#     plt.plot(x, domain_test_preds[name], 'y', label='domain')
#     plt.yscale("log")
#     plt.title(name)
#     plt.xlabel('wip')
#     plt.ylabel('latency')
#     plt.legend()
#     # plt.show()
#     plt.savefig('../Plots/forecasts/' + name.replace("/", "_") + '_test.png')
#     plt.close()


df = df.groupby(by="api_name")

for name, group in df:

    results_file.write(name)
    results_file.write("\n")
    results_file.write(str(regression_test_preds[name]))
    results_file.write("\n")

    x = np.arange(0, group.wip.max() +1 , 0.1)

    plt.scatter(group.wip, group.latency)
    plt.plot(x, regression_test_preds[name], 'r', label='regression')
    plt.plot(np.arange(0, int(group.wip.max()) + 1, 0.1), xgboost_test_preds[name], 'm', label='xgboost')
    plt.plot(x, domain_test_preds[name], 'y', label='domain')
    plt.yscale("log")
    plt.title(name)
    plt.xlabel('wip')
    plt.ylabel('latency')
    plt.legend()
    # plt.show()
    plt.savefig('../Plots/forecasts/' + name.replace("/", "_") + '.png')
    plt.close()

results_file.close()