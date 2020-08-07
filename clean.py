# -*- coding: utf-8 -*-
"""
Cleans up a data set to have no missing values and only numerical entries

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


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

# export the data
X.to_csv("X clean.csv", index=False)
Y.to_csv("Y clean.csv", index=False)
