# -*- coding: utf-8 -*-
"""
Tunes a K Nearest Neighbors model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
    model = KNeighborsClassifier(n_neighbors=5, weights="uniform",
                                 leaf_size=30, n_jobs=1)
else:
    model = KNeighborsRegressor(n_neighbors=5, weights="uniform",
                                leaf_size=30, n_jobs=1)

# set up the tuner
parameters = {"n_neighbors": [3, 5, 10, 20],
              "leaf_size": [10, 20, 30, 40]}
grid = RandomizedSearchCV(model, parameters, n_iter=16, cv=3, 
                          random_state=0, n_jobs=None)

# tune the model
search = grid.fit(X, Y)

# export results
results = pd.DataFrame(search.best_params_, index=[0])
results.to_csv("tune knn.csv", index=False)
