# -*- coding: utf-8 -*-
"""
Cleans up a data set to have no missing values, no outliers, and only numbers
Adds 2nd order polynomial features

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PolynomialFeatures


# read in the data
X = pd.read_csv("X.csv")
Y = pd.read_csv("Y.csv")

# determine which columns are strings (for X)
x_dtypes = X.dtypes
x_str = np.where(x_dtypes == "object")[0]

# determine which columns are strings (for Y)
y_dtypes = Y.dtypes
y_str = np.where(y_dtypes == "object")[0]

# fill in missing values with the most frequent column value (for X)
impute = SimpleImputer(strategy="most_frequent")
x_columns = X.columns
X = pd.DataFrame(data=impute.fit_transform(X), columns=x_columns)

# fill in missing values with the most frequent column value (for Y)
y_columns = Y.columns
Y = pd.DataFrame(impute.fit_transform(Y), columns=y_columns)

# convert any string columns to binary columns
X = pd.get_dummies(X, columns=x_columns[x_str])
Y = pd.get_dummies(Y, columns=y_columns[y_str])

# train a model to detect outliers
data = pd.concat([Y, X], axis=1)
model = LocalOutlierFactor(n_neighbors=20, leaf_size=30, n_jobs=1)
model.fit(data)

# remove 2% of the data
percent = 0.02
cutoff = np.quantile(model.negative_outlier_factor_, percent)
good_idx = np.where(model.negative_outlier_factor_ > cutoff)[0]
X = X.iloc[good_idx, :].reset_index(drop=True)
Y = Y.iloc[good_idx, :].reset_index(drop=True)

# add 2nd order polynomial features to X
poly = PolynomialFeatures(2, include_bias=False)
x_columns = X.columns
X = pd.DataFrame(poly.fit_transform(X))
X.columns = poly.get_feature_names(x_columns)

# export the data
X.to_csv("X clean.csv", index=False)
Y.to_csv("Y clean.csv", index=False)
