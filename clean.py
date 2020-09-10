# -*- coding: utf-8 -*-
"""
Cleans up a data set to have no missing values and only numbers

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


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

# get the text columns from the data
text_x = X.iloc[:, x_str]
text_y = Y.iloc[:, y_str]
X = X.drop(columns=x_columns[x_str])
Y = Y.drop(columns=y_columns[y_str])

# collect the words (and their inverse frequencies) from each document
# 'matrix_x' is a term (columns) document (rows) matrix
matrix_x = pd.DataFrame()
for c in text_x.columns:
    vector = TfidfVectorizer()
    matrix2 = vector.fit_transform(text_x[c].tolist())
    names = vector.get_feature_names()
    names = [str(c) + " " + str(n) for n in names]
    matrix2 = pd.DataFrame(matrix2.toarray(), columns=names)
    matrix_x = pd.concat([matrix_x, matrix2], axis=1)

# 'matrix_y' is a term (columns) document (rows) matrix
matrix_y = pd.DataFrame()
for c in text_y.columns:
    vector = TfidfVectorizer()
    matrix2 = vector.fit_transform(text_y[c].tolist())
    names = vector.get_feature_names()
    names = [str(c) + " " + str(n) for n in names]
    matrix2 = pd.DataFrame(matrix2.toarray(), columns=names)
    matrix_y = pd.concat([matrix_y, matrix2], axis=1)

# combine the data
X = pd.concat([X, matrix_x], axis=1)
Y = pd.concat([Y, matrix_y], axis=1)

# export the data
X.to_csv("X clean.csv", index=False)
Y.to_csv("Y clean.csv", index=False)
