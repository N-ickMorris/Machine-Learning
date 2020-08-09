# -*- coding: utf-8 -*-
"""
Trains and tests a k-means clustering model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from plots import pairs_plot


# read in the data
X = pd.read_csv("X clean.csv")

# standardize the data to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# separate the data into training and testing
np.random.seed(1)
test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# train a k-means model
cluster = KMeans(n_clusters=6, n_init=20, max_iter=300, tol=0.0001, random_state=42,
                 n_jobs=1)
cluster.fit(X.iloc[train_idx, :])

# compute clusters for all the data
labels = cluster.predict(X)

# train a PCA model
component = PCA(n_components=3, random_state=42)
component.fit(X.iloc[train_idx, :])

# compute components for all the data, add cluster labels and train/test labels
components = pd.DataFrame(component.transform(X), columns=["PC1", "PC2", "PC3"])
components["Cluster"] = labels
components["Data"] = "Train"
for j in test_idx:
    components.loc[j, "Data"] = "Test"
components.to_csv("kmeans and pca.csv", index=False)

# tells how well separated the clusters are
train_score = str(np.round(silhouette_score(X.iloc[train_idx, :],
                                            components.loc[train_idx, "Cluster"]), 3))
test_score = str(np.round(silhouette_score(X.iloc[test_idx, :],
                                           components.loc[test_idx, "Cluster"]), 3))

# plot the clusters
save_plot = False
pairs_plot(components.iloc[train_idx,:], vars=components.columns[:3],
           color="Cluster", title="K-Means Clustering - Train - Silhouette: " + train_score,
           save=save_plot)
pairs_plot(components.iloc[train_idx,:], vars=components.columns[:3],
           color="Cluster", title="K-Means Clustering - Test - Silhouette: " + test_score,
           save=save_plot)
