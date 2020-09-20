# -*- coding: utf-8 -*-
"""
Neural Network Pipeline

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, r2_score


TIME_SERIES = False

# read in the data
X = pd.read_csv("X clean.csv")
Y = pd.read_csv("Y clean.csv")

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])

# set up the pipeline
if classifier:
    pipeline = Pipeline([
        ('var', VarianceThreshold()),
        ('scale', MinMaxScaler()),
        ('model', MLPClassifier(max_iter=50, activation="relu", solver="adam", 
                                learning_rate="adaptive", random_state=42)),
    ])
else:
    pipeline = Pipeline([
        ('var', VarianceThreshold()),
        ('scale', MinMaxScaler()),
        ('model', MLPRegressor(max_iter=50, activation="relu", solver="adam", 
                               learning_rate="adaptive",random_state=42)),
    ])

# set up the grid search
parameters = {
    'model__hidden_layer_sizes': ((32, 32), (64, 64), (128, 128)),
    'model__batch_size': (16, 32),
    'model__learning_rate_init': (0.0001, 0.001, 0.01),
}
grid_search = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1, verbose=1)

# separate the data into training and testing
if TIME_SERIES:
    test_idx = X.index.values[-int(X.shape[0] / 5):]
else:
    np.random.seed(1)
    test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# search the grid for the best the model
grid_search.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])
model = grid_search.best_estimator_

# predict training and testing data
train_predict = pd.DataFrame(model.predict(X.iloc[train_idx, :]), columns=Y.columns)
test_predict = pd.DataFrame(model.predict(X.iloc[test_idx, :]), columns=Y.columns)

# collect actual data
train_actual = Y.iloc[train_idx, :].copy().reset_index(drop=True)
test_actual = Y.iloc[test_idx, :].copy().reset_index(drop=True)

# score the goodness of fit
train_score = []
test_score = []
for j in range(train_predict.shape[1]):
    if classifier:
        train_score.append(accuracy_score(train_actual.iloc[:,j], train_predict.iloc[:,j]))
        test_score.append(accuracy_score(test_actual.iloc[:,j], test_predict.iloc[:,j]))
    else:
        train_score.append(r2_score(train_actual.iloc[:,j], train_predict.iloc[:,j]))
        test_score.append(r2_score(test_actual.iloc[:,j], test_predict.iloc[:,j]))
train_score = np.mean(train_score)
test_score = np.mean(test_score)

# report the scores
if classifier:
    print("Train Accuracy: " + str(train_score) + "\n"
          "Test Accuracy: " + str(test_score))
else:
    print("Train R2: " + str(train_score) + "\n"
          "Test R2: " + str(test_score))