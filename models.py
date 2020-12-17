from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model

def create_model(dim):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(6, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="linear"))

    return model

def create_residual_model(dim):
    model = Sequential()
    model.add(Dense(6, input_dim=dim, activation="relu"))
    # model.add(Dense(6, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="linear"))

    return model

def create_regularization_model(dim):
    inp = Input((dim,))
    x = Lambda(lambda x: x[:,:-1])(inp)
    o2 = Lambda(lambda x: x[:,-1:])(inp)
    # x = Dense(8, activation="relu")(x)
    # x = Dense(6, activation="relu")(x)
    x = Dense(4, activation="relu")(x)
    o1 = Dense(1, activation="linear")(x)
    out = concatenate([o1, o2])

    model = Model(inp, out)

    return model