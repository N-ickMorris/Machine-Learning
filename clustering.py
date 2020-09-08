# -*- coding: utf-8 -*-
"""
Creates Hierarchical Clusters

@author: Nick
"""


import pandas as pd
from sklearn.cluster import AgglomerativeClustering


# read in the data
X_copy = pd.read_csv("X clean.csv")

# standardize the data to take on values between 0 and 1
X = ((X_copy - X_copy.min()) / (X_copy.max() - X_copy.min())).copy()

# create clusters
num = 6
hclust = AgglomerativeClustering(n_clusters=num, linkage="ward", distance_threshold=None)
hclust.fit(X)
labels = pd.DataFrame(hclust.labels_, columns=["Cluster"])

# convert any string columns to binary columns
labels["Cluster"] = labels["Cluster"].astype(str)
labels = pd.get_dummies(labels)

# export the data
X = pd.concat([X_copy, labels], axis=1)
X.to_csv("X clean.csv", index=False)
