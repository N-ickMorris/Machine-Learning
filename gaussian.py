# -*- coding: utf-8 -*-
"""
Trains and tests a Gaussian Process model on data

@author: Nick
"""


import re
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


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

# set up the model
if classifier:
    model = MultiOutputClassifier(GaussianProcessClassifier(random_state=42,
                                                            n_jobs=1))
else:
    model = MultiOutputRegressor(GaussianProcessRegressor(random_state=42,
                                                          n_jobs=1))

# train the model
model.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

# In[2]: Collect the predictions

# predict training and testing data
train_predict = pd.DataFrame(model.predict(X.iloc[train_idx, :]), columns=Y.columns)
test_predict = pd.DataFrame(model.predict(X.iloc[test_idx, :]), columns=Y.columns)

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
predictions.to_csv("gaussian predictions", index=False)

# In[3]: Visualize the predictions

def matrix_plot(matrix, title=" ", save=False):
    # set up labels for the plot
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         matrix.flatten()/np.sum(matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    # plot the predictions
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=labels, fmt="", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predict")
    ax.set_ylabel("Actual")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def parity_plot(predict, actual, title=" ", alpha=2/3, save=False):
    # plot the predictions
    fig, ax = plt.subplots()
    sns.scatterplot(predict, actual, color="blue", alpha=alpha, ax=ax)
    sns.lineplot(actual, actual, color="red", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predict")
    ax.set_ylabel("Actual")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

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