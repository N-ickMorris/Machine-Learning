# -*- coding: utf-8 -*-
"""
Tunes a Support Vector Machine model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.model_selection import RandomizedSearchCV


# read in the data
X = pd.read_csv("X titanic.csv")
Y = pd.read_csv("Y titanic.csv").iloc[:,[0]]

# standardize the inputs to take on values between 0 and 1
x_columns = X.columns
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=x_columns)

# determine if we are building a classifier model
classifier = np.all(np.unique(Y.to_numpy()) == [0, 1])

# set up the model
if classifier:
    model = LinearSVC(C=1, class_weight="balanced", random_state=42)
    parameters = {"C": [0.001, 0.01, 0.5, 1, 1.5, 2]}
else:
    model = LinearSVR(epsilon=0, C=1, random_state=42)
    parameters = {"epsilon": [0, 0.001, 0.01, 0.1],
                  "C": [0.001, 0.01, 0.5, 1, 1.5, 2]}

# set up the tuner
grid = RandomizedSearchCV(model, parameters, n_iter=16, cv=3, 
                          random_state=0, n_jobs=None)

# tune the model
search = grid.fit(X, Y)

# export results
results = pd.DataFrame(search.best_params_, index=[0])
# results.to_csv("tune svm.csv", index=False)
