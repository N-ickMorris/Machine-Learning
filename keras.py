# -*- coding: utf-8 -*-
"""
Trains and tests a Tensorflow Neural Network model on data

@author: Nick
"""


import numpy as np
import pandas as pd
import keras
from keras import layers, optimizers, regularizers
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from plots import matrix_plot, parity_plot


# In[1]: Train the model

# read in the data
X = pd.read_csv("X clean.csv")
Y = pd.read_csv("Y clean.csv")

# standardize the inputs to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])
outputs = Y.shape[1]

# separate the data into training and testing
np.random.seed(1)
test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# set up the network
def build_nnet(features, targets, layer=[32, 32], learning_rate=0.001, l1_penalty=0, classifier=False):
    # set up the output layer activation and loss metric
    if classifier:
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "linear"
        loss = "mean_squared_error"

    # build the network
    inputs = keras.Input(shape=(features,))
    hidden = layers.Dense(units=layer[0], activation="relu",
                          kernel_regularizer=regularizers.l1(l1_penalty))(inputs)
    for j in range(1, len(layer)):
        dense = layers.Dense(units=layer[j], activation="relu",
                             kernel_regularizer=regularizers.l1(l1_penalty))
        hidden = dense(hidden)
    outputs = layers.Dense(units=targets, activation=activation)(hidden)

    # compile the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss=[loss for j in layer],
                  optimizer=optimizers.Adam(lr=learning_rate))
    return model

# set up the model
if classifier:
    model = build_nnet(features=X.shape[1], targets=Y.shape[1], layer=[32, 32],
                       learning_rate=0.001, l1_penalty=0, classifier=True)
else:
    model = build_nnet(features=X.shape[1], targets=Y.shape[1], layer=[32, 32],
                       learning_rate=0.001, l1_penalty=0, classifier=False)

# train the model
model.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :], epochs=100, batch_size=16)

# In[2]: Collect the predictions

# predict training and testing data
train_predict = pd.DataFrame(model.predict(X.iloc[train_idx, :]), columns=Y.columns)
test_predict = pd.DataFrame(model.predict(X.iloc[test_idx, :]), columns=Y.columns)

# convert probabilities into predictions
train_predict = (train_predict > 0.5) + 0
test_predict = (test_predict > 0.5) + 0

# reshape all of the predictions into a single table
predictions = pd.DataFrame()
for j in range(outputs):
    # collect training data
    predict_j = np.array(train_predict.iloc[:,j])
    actual_j = np.array(Y.iloc[train_idx, j])
    name_j = Y.columns[j]
    data_j = "Train"
    predictions = pd.concat([predictions,
                            pd.DataFrame({"Predict": predict_j,
                                          "Actual": actual_j,
                                          "Name": np.repeat(name_j,
                                                            len(train_idx)),
                                          "Data": np.repeat(data_j,
                                                            len(train_idx))})],
                            axis="index")

    # collect testing data
    predict_j = np.array(test_predict.iloc[:,j])
    actual_j = np.array(Y.iloc[test_idx, j])
    name_j = Y.columns[j]
    data_j = "Test"
    predictions = pd.concat([predictions,
                            pd.DataFrame({"Predict": predict_j,
                                          "Actual": actual_j,
                                          "Name": np.repeat(name_j,
                                                            len(test_idx)),
                                          "Data": np.repeat(data_j,
                                                            len(test_idx))})],
                            axis="index")
predictions = predictions.reset_index(drop=True)
predictions.to_csv("keras predictions.csv", index=False)

# In[3]: Visualize the predictions

# confusion matrix heatmap
save_plot = False
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
