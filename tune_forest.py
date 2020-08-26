# -*- coding: utf-8 -*-
"""
Tunes a Random Forest model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=14,
                                   min_samples_leaf=5,
                                   max_features="sqrt",
                                   class_weight="balanced_subsample",
                                   random_state=42,
                                   n_jobs=1)
else:
    model = RandomForestRegressor(n_estimators=100,
                                  max_depth=14,
                                  min_samples_leaf=5,
                                  max_features="sqrt",
                                  random_state=42,
                                  n_jobs=1)

# set up the tuner
parameters = {"n_estimators": [25, 50, 100],
              "max_depth": [6, 10, 14, 18],
              "min_samples_leaf": [1, 3, 5, 10]}
grid = RandomizedSearchCV(model, parameters, n_iter=16, cv=3, 
                          random_state=0, n_jobs=None)

# tune the model
search = grid.fit(X, Y)

# export results
results = pd.DataFrame(search.best_params_, index=[0])
results.to_csv("tune forest.csv", index=False)
