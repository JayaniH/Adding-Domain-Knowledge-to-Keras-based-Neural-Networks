from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

cs = MinMaxScaler()


def load_data():
    df = pd.read_csv('performance_data_truncated.csv', sep="\t")
    df = df[df.wip < 1500]
    df = df.groupby(by="api_name")

    return df

load_data()

def process_attributes(train, test):
    continuous = ["wip"]
    # cs = MinMaxScaler()
    trainX = cs.fit_transform(train[continuous])
    testX = cs.transform(test[continuous])

    return (trainX, testX)

def scale_x(x):
    # cs = MinMaxScaler()
    x = cs.fit_transform(x)
    # print(x)
    return x

def scale_y(y):
    # cs = MinMaxScaler()
    y = cs.fit(y)

    return y

def undo_scale_y(y):
    # cs = MinMaxScaler()
    y = cs.inverse_transform(y)

    return y