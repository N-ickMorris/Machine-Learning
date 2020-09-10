# -*- coding: utf-8 -*-
"""
Creates 2nd order polynomial features
Selects unique features using Hierarchical Clustering

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.cluster import FeatureAgglomeration
from plots import plot_dendrogram, corr_plot


# should we include division in the polynomials?
DIVISION = False

# read in the data
X_copy = pd.read_csv("X clean.csv")

# add reciprocal features if desired
if DIVISION:
    X_recip = (1 / X_copy).copy()
    X_recip.columns = ["recip_" + str(c) for c in X_recip.columns]
    X_recip = X_recip.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    X_copy = pd.concat([X_copy, X_recip], axis=1)
    del X_recip

# add 2nd order polynomial features to X
poly = PolynomialFeatures(2, include_bias=False)
x_columns = X_copy.columns
X_copy = pd.DataFrame(poly.fit_transform(X_copy))
X_copy.columns = poly.get_feature_names(x_columns)

# drop any constant columns in X
X_copy = X_copy.loc[:, (X_copy != X_copy.iloc[0]).any()]

# standardize the inputs to take on values between 0 and 1
x_columns = X_copy.columns
scaler = MinMaxScaler()
X = scaler.fit_transform(X_copy)
X = pd.DataFrame(X, columns=x_columns)

# plot the correlations in the data to visualize clusters
corr_plot(X, method="ward")

# plot the model to see how many features to keep
hclust = FeatureAgglomeration(n_clusters=None, linkage="ward", distance_threshold=0)
hclust.fit(X)
plot_dendrogram(hclust)

# build the final model
num = int(X.shape[1] / 4)
hclust = FeatureAgglomeration(n_clusters=num, linkage="ward", distance_threshold=None)
hclust.fit(X)

# collect the features to keep
clusters = hclust.labels_
keep = []
for i in range(num):
    keep.append(np.where(clusters == i)[0][0])
X = X_copy.iloc[:, keep]

# export the data
X.to_csv("X clean.csv", index=False)
