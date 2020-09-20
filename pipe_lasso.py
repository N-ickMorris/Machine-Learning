# -*- coding: utf-8 -*-
"""
Polynomial Lasso Pipeline

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegressionCV, LassoCV
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import accuracy_score, r2_score


TIME_SERIES = False

# read in the data
X = pd.read_csv("X clean.csv")
Y = pd.read_csv("Y clean.csv")

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])

# max features in the model
max_features = 100

if classifier:
    pipeline = Pipeline([
        ('var1', VarianceThreshold()),
        ('poly', PolynomialFeatures(2)),
        ('var2', VarianceThreshold()),
        ('scale', MinMaxScaler()),
        ('select', SelectFromModel(RandomForestClassifier(n_estimators=50,
                                                          max_depth=10,
                                                          min_samples_leaf=1,
                                                          max_features="sqrt",
                                                          class_weight="balanced_subsample",
                                                          random_state=42,
                                                          n_jobs=-1), max_features=max_features)),
        ('model', MultiOutputClassifier(LogisticRegressionCV(penalty="l1", solver="saga",
                                                             Cs=16, cv=3, tol=1e-4,
                                                             max_iter=100,
                                                             class_weight="balanced",
                                                             random_state=42,
                                                             n_jobs=-1))),
    ], verbose=True)
else:
    pipeline = Pipeline([
        ('var1', VarianceThreshold()),
        ('poly', PolynomialFeatures(2)),
        ('var2', VarianceThreshold()),
        ('scale', MinMaxScaler()),
        ('select', SelectFromModel(RandomForestRegressor(n_estimators=50,
                                                         max_depth=10,
                                                         min_samples_leaf=1,
                                                         max_features="sqrt",
                                                         random_state=42,
                                                         n_jobs=-1), max_features=max_features)),
        ('model', MultiOutputRegressor(LassoCV(eps=1e-9, n_alphas=16, cv=3,
                                               tol=1e-4, max_iter=100, random_state=42,
                                               n_jobs=-1))),
    ], verbose=True)

# separate the data into training and testing
if TIME_SERIES:
    test_idx = X.index.values[-int(X.shape[0] / 5):]
else:
    np.random.seed(1)
    test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# train the model
pipeline.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

# predict training and testing data
train_predict = pd.DataFrame(pipeline.predict(X.iloc[train_idx, :]), columns=Y.columns)
test_predict = pd.DataFrame(pipeline.predict(X.iloc[test_idx, :]), columns=Y.columns)

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