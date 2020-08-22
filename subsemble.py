# -*- coding: utf-8 -*-
"""
Trains and tests a Subsemble model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from mlens.ensemble import Subsemble
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.metrics import confusion_matrix, accuracy_score, r2_score
from plots import matrix_plot, parity_plot, series_plot

TIME_SERIES = False

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
if TIME_SERIES:
    test_idx = X.index.values[-int(X.shape[0] / 5):]
else:
    np.random.seed(1)
    test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# set up the model
if classifier:
    model = Subsemble(partitions=2, random_state=42, n_jobs=1)
    model.add(LogisticRegression(penalty="l1", solver="saga")).add(KNeighborsClassifier()).add(RandomForestClassifier()).add(LinearSVC()).add(GaussianNB()).add(GaussianProcessClassifier()).add_meta(RandomForestClassifier())
else:
    model = Subsemble(partitions=2, random_state=42, n_jobs=1)
    model.add(Lasso()).add(KNeighborsRegressor()).add(RandomForestRegressor()).add(LinearSVR()).add(BayesianRidge()).add(GaussianProcessRegressor()).add_meta(RandomForestRegressor())

# train and predict
train_predict = pd.DataFrame()
test_predict = pd.DataFrame()
for j in Y.columns:
    # train the model
    model.fit(X.iloc[train_idx, :], Y.loc[train_idx, j])

    # predict training and testing data
    train_predict_j = pd.DataFrame(model.predict(X.iloc[train_idx, :]), columns=[j])
    test_predict_j = pd.DataFrame(model.predict(X.iloc[test_idx, :]), columns=[j])
    train_predict = pd.concat([train_predict, train_predict_j], axis=1)
    test_predict = pd.concat([test_predict, test_predict_j], axis=1)
    print("-- " + j + " --")


# In[2]: Collect the predictions

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
predictions.to_csv("subsemble predictions.csv", index=False)

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

        if TIME_SERIES:
            # series plot
            series_plot(predict=data["Predict"], actual=data["Actual"], 
                        title=j + " - " + data_j + " - Forecast: ", save=save_plot)
