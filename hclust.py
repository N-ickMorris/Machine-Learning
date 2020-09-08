# -*- coding: utf-8 -*-
"""
Trains a Hierarchical Clustering model on data

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from plots import pairs_plot


# read in the data
X = pd.read_csv("X clean.csv")

# standardize the data to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# train a Hierarchical Clustering model
cluster = AgglomerativeClustering(n_clusters=6, linkage="ward", distance_threshold=None)
labels = cluster.fit_predict(X)

# train a PCA model
n_comp = 3 # number of principal components
component = PCA(n_components=n_comp, random_state=42)
component.fit(X)

# compute components for all the data, add cluster labels and train/test labels
components = pd.DataFrame(component.transform(X), 
                          columns=["PC" + str(i + 1) for i in range(n_comp)])
components["Cluster"] = labels
components["Data"] = "Train"
components.to_csv("hclust and pca.csv", index=False)

# tells how well separated the clusters are
train_score = str(np.round(silhouette_score(X,
                                            components.loc[:, "Cluster"]), 3))

# plot the clusters
save_plot = False
pairs_plot(components, vars=components.columns[:n_comp],
           color="Cluster", title="Hierarchical Clustering & PCA - Silhouette: " + train_score,
           save=save_plot)

# train a random forest to learn the clusters
model = RandomForestClassifier(n_estimators=50, max_depth=10,
                               min_samples_leaf=5, max_features="sqrt",
                               class_weight="balanced_subsample",
                               random_state=42, n_jobs=1)
model.fit(X, labels)

# collect and sort feature importance
importance = pd.DataFrame({"name": X.columns,
                           "importance": model.feature_importances_})
importance = importance.sort_values(by="importance", ascending=False).reset_index(drop=True)

# choose how many features to plot
num_features = 3
df = pd.concat([X, pd.DataFrame(labels, columns=["cluster"])], axis=1)
features = importance.loc[:(num_features - 1), "name"].tolist()

# plot the variables
pairs_plot(df, vars=features,
           color="cluster", title="Hierarchical Clustering - Silhouette: " + train_score,
           save=save_plot)
