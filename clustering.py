# -*- coding: utf-8 -*-
"""
Creates k-Means Clusters

@author: Nick
"""


import pandas as pd
from sklearn.cluster import KMeans


# read in the data
X_copy = pd.read_csv("X clean.csv")

# standardize the data to take on values between 0 and 1
X = ((X_copy - X_copy.min()) / (X_copy.max() - X_copy.min())).copy()

# train a k-means model
cluster = KMeans(n_clusters=6, n_init=20, max_iter=300, tol=0.0001, random_state=42,
                 n_jobs=1)
cluster.fit(X)

# compute clusters for all the data
labels = pd.DataFrame(cluster.predict(X), columns=["Cluster"])

# convert any string columns to binary columns
labels["Cluster"] = labels["Cluster"].astype(str)
labels = pd.get_dummies(labels)

# export the data
X = pd.concat([X_copy, labels], axis=1)
X.to_csv("X clean.csv", index=False)
