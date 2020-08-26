# -*- coding: utf-8 -*-
"""
Tunes an Extreme Gradient Boosting Tree model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


# read in the data
X = pd.read_csv("X arrest.csv")
Y = pd.read_csv("Y arrest.csv").iloc[:,[0]]

# standardize the inputs to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])

# set up the model
if classifier:
    num_zeros = len(np.where(Y.to_numpy() == 0)[0])
    num_ones = Y.shape[0] - num_zeros
    model = XGBClassifier(booster="gbtree",
                          n_estimators=100, learning_rate=0.1,
                          max_depth=7,
                          min_child_weight=1,
                          colsample_bytree=0.8,
                          subsample=0.8,
                          scale_pos_weight = num_zeros / num_ones,
                          random_state=42,
                          n_jobs=1)
else:
    model = XGBRegressor(booster="gbtree",
                         n_estimators=100, learning_rate=0.1,
                         max_depth=7,
                         min_child_weight=1,
                         colsample_bytree=0.8,
                         subsample=0.8,
                         random_state=42,
                         n_jobs=1)

# set up the tuner
parameters = {"n_estimators": [50],
              "learning_rate": [0.001, 0.01, 0.1, 1],
              "max_depth": [3, 6, 9, 12],
              "min_child_weight": [1, 3, 5],
              "colsample_bytree": [0.8],
              "subsample": [0.8]}
grid = RandomizedSearchCV(model, parameters, n_iter=16, cv=3,
                          random_state=0, n_jobs=None)

# tune the model
search = grid.fit(X, Y)

# export results
results = pd.DataFrame(search.best_params_, index=[0])
results.to_csv("tune xgboost.csv", index=False)
