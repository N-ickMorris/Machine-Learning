# -*- coding: utf-8 -*-
"""
Trains and tests a Tensorflow Long Short Term Memory Neural Network model on data

@author: Nick
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras import layers, optimizers, regularizers
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from plots import matrix_plot, parity_plot, series_plot


# In[1]: Train the model

# load dataset
X = pd.read_csv('X time.csv')
Y = pd.read_csv('Y time.csv')

# standardize the inputs to take on values between 0 and 1
x_columns = X.columns
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=x_columns)

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])
outputs = Y.shape[1]

# split into train and test sets
values_X = X.values
values_y = Y.values
train_samples = int(0.8 * X.shape[0])
train_X = values_X[:train_samples, :]
test_X = values_X[train_samples:, :]
train_y = values_y[:train_samples, :]
test_y = values_y[train_samples:, :]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# set up the network
def build_nnet(features, targets, timesteps=1, layer=[32, 32], learning_rate=0.001, l1_penalty=0, classifier=False):
    # set up the output layer activation and loss metric
    if classifier:
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "linear"
        loss = "mean_squared_error"

    # build the network
    inputs = keras.Input(shape=(timesteps, features))
    hidden = layers.LSTM(units=layer[0], activation="relu",
                         kernel_regularizer=regularizers.l1(l1_penalty),
                         return_sequences=True)(inputs)
    for j in range(1, len(layer) - 1):
        recurrent = layers.LSTM(units=layer[j], activation="relu",
                                kernel_regularizer=regularizers.l1(l1_penalty),
                                return_sequences=True)
        hidden = recurrent(hidden)
    recurrent = layers.LSTM(units=layer[-1], activation="relu",
                            kernel_regularizer=regularizers.l1(l1_penalty))
    hidden = recurrent(hidden)
    outputs = layers.Dense(units=targets, activation=activation)(hidden)

    # compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=[loss for j in layer],
                  optimizer=optimizers.Adam(lr=learning_rate))
    return model

# set up the model
if classifier:
    num_zeros = len(np.where(Y.to_numpy() == 0)[0])
    num_ones = Y.shape[0] - num_zeros
    class_weights = {0: Y.shape[0] / num_zeros,
                     1: Y.shape[0] / num_ones}
    model = build_nnet(features=X.shape[1], targets=Y.shape[1], timesteps=1,
                       layer=[32, 32], learning_rate=0.001, l1_penalty=0, classifier=True)
else:
    class_weights = None
    model = build_nnet(features=X.shape[1], targets=Y.shape[1], timesteps=1,
                   layer=[32, 32], learning_rate=0.001, l1_penalty=0, classifier=False)

# train the model
model.fit(train_X, train_y, epochs=100, batch_size=16, class_weight=class_weights)

# In[2]: Collect the predictions

# predict training and testing data
train_predict = pd.DataFrame(model.predict(train_X), columns=Y.columns)
test_predict = pd.DataFrame(model.predict(test_X), columns=Y.columns)

# convert probabilities into predictions
if classifier:
    train_predict = (train_predict > 0.5) + 0
    test_predict = (test_predict > 0.5) + 0

# reshape all of the predictions into a single table
predictions = pd.DataFrame()
for j in range(outputs):
    # collect training data
    predict_j = np.array(train_predict.iloc[:,j])
    actual_j = np.array(train_y[:, j])
    name_j = Y.columns[j]
    data_j = "Train"
    predictions = pd.concat([predictions,
                            pd.DataFrame({"Predict": predict_j,
                                          "Actual": actual_j,
                                          "Name": np.repeat(name_j,
                                                            train_y.shape[0]),
                                          "Data": np.repeat(data_j,
                                                            train_y.shape[0])})],
                            axis="index")

    # collect testing data
    predict_j = np.array(test_predict.iloc[:,j])
    actual_j = np.array(test_y[:, j])
    name_j = Y.columns[j]
    data_j = "Test"
    predictions = pd.concat([predictions,
                            pd.DataFrame({"Predict": predict_j,
                                          "Actual": actual_j,
                                          "Name": np.repeat(name_j,
                                                            test_y.shape[0]),
                                          "Data": np.repeat(data_j,
                                                            test_y.shape[0])})],
                            axis="index")
predictions = predictions.reset_index(drop=True)
predictions.to_csv("lstm predictions.csv", index=False)

# In[3]: Visualize the predictions

# confusion matrix heatmap
save_plot = True
if classifier:
    for j in Y.columns:
        # training data
        data_j = "Train"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        accuracy = str(np.round(accuracy_score(data["Actual"],
                                               data["Predict"]) * 100, 1)) + "%"
        matrix = confusion_matrix(data["Actual"], data["Predict"])
        matrix_plot(matrix, title=j + " - " + data_j + " - Accuracy: " + accuracy,
                    save=save_plot)

        # testing data
        data_j = "Test"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        accuracy = str(np.round(accuracy_score(data["Actual"],
                                               data["Predict"]) * 100, 1)) + "%"
        matrix = confusion_matrix(data["Actual"], data["Predict"])
        matrix_plot(matrix, title=j + " - " + data_j + " - Accuracy: " + accuracy,
                    save=save_plot)

# parity plot
else:
    for j in Y.columns:
        # training data
        data_j = "Train"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        r2 = str(np.round(r2_score(data["Actual"], data["Predict"]) * 100, 1)) + "%"
        parity_plot(predict=data["Predict"], actual=data["Actual"],
                    title=j + " - " + data_j + " - R2: " + r2, save=save_plot)

        # testing data
        data_j = "Test"
        data = predictions.loc[(predictions["Data"] == data_j) & (predictions["Name"] == j)]
        r2 = str(np.round(r2_score(data["Actual"], data["Predict"]) * 100, 1)) + "%"
        parity_plot(predict=data["Predict"], actual=data["Actual"],
                    title=j + " - " + data_j + " - R2: " + r2, save=save_plot)

        # series plot
        series_plot(predict=data["Predict"], actual=data["Actual"], 
                    title=j + " - " + data_j + " - Forecast: ", save=save_plot)