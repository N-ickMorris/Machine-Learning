# -*- coding: utf-8 -*-
"""
Cleans up a data set to have no outliers

@author: Nick
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from plots import pairs_plot


# read in the data
X = pd.read_csv("X clean.csv")
Y = pd.read_csv("Y clean.csv")

# train a model to detect outliers
data = pd.concat([Y, X], axis=1)
model = LocalOutlierFactor(n_neighbors=20, leaf_size=30, novelty=False, n_jobs=1)
model.fit(data)

# determine how much of the data has outliers
percent = 0.1
cutoff = np.quantile(model.negative_outlier_factor_, percent)
labels = (model.negative_outlier_factor_ > cutoff) * 1

# train a PCA model
n_comp = 3 # number of principal components
component = PCA(n_components=n_comp, random_state=42)
component.fit(X)

# compute components for all the data, add cluster labels and train/test labels
components = pd.DataFrame(component.transform(X), 
                          columns=["PC" + str(i + 1) for i in range(n_comp)])
components["Inlier"] = labels
components["Data"] = "Train"
components.to_csv("inliers and pca.csv", index=False)

# tells how well separated the clusters are
train_score = str(np.round(silhouette_score(X,
                                            components.loc[:, "Inlier"]), 3))

# plot the clusters
save_plot = False
pairs_plot(components, vars=components.columns[:n_comp],
           color="Inlier", title="Local Outlier Factor & PCA - Silhouette: " + train_score,
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
df = pd.concat([X, pd.DataFrame(labels, columns=["Inlier"])], axis=1)
features = importance.loc[:(num_features - 1), "name"].tolist()

# plot the variables
pairs_plot(df, vars=features,
           color="Inlier", title="Local Outlier Factor - Silhouette: " + train_score,
           save=save_plot)

# remove the outliers
good_idx = np.where(model.negative_outlier_factor_ > cutoff)[0]
X = X.iloc[good_idx, :].reset_index(drop=True)
Y = Y.iloc[good_idx, :].reset_index(drop=True)

# export the data
X.to_csv("X clean.csv", index=False)
Y.to_csv("Y clean.csv", index=False)
