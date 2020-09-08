# -*- coding: utf-8 -*-
"""
Creates 2nd order polynomial features
Selects unique features using Hierarchical Clustering

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import FeatureAgglomeration
from plots import plot_dendrogram

# read in the data
X_copy = pd.read_csv("X clean.csv")

# add 2nd order polynomial features to X
poly = PolynomialFeatures(2, include_bias=False)
x_columns = X_copy.columns
X_copy = pd.DataFrame(poly.fit_transform(X_copy))
X_copy.columns = poly.get_feature_names(x_columns)

# drop any constant columns in X
X_copy = X_copy.loc[:, (X_copy != X_copy.iloc[0]).any()]

# standardize the data to take on values between 0 and 1
X = ((X_copy - X_copy.min()) / (X_copy.max() - X_copy.min())).copy()

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
