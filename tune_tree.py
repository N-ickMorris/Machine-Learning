# -*- coding: utf-8 -*-
"""
Tunes a Decision Tree model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV


# read in the data
X = pd.read_csv("X clean.csv")
Y = pd.read_csv("Y clean.csv").iloc[:,[0]]

# standardize the inputs to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])

# set up the model
if classifier:
    model = DecisionTreeClassifier(max_depth=14,
                                   min_samples_leaf=5,
                                   max_features="sqrt",
                                   random_state=42)
else:
    model = DecisionTreeRegressor(max_depth=14,
                                  min_samples_leaf=5,
                                  max_features="sqrt",
                                  random_state=42)

# set up the tuner
parameters = {"max_depth": [6, 10, 14, 18],
              "min_samples_leaf": [1, 3, 5, 10]}
grid = RandomizedSearchCV(model, parameters, n_iter=16, cv=3, 
                          random_state=0, n_jobs=None)

# tune the model
search = grid.fit(X, Y)

# export results
results = pd.DataFrame(search.best_params_, index=[0])
results.to_csv("tune tree.csv", index=False)
