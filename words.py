# -*- coding: utf-8 -*-
"""
Clusters text with k-Means and predicts clusters with Random Forest

@author: Nick
"""


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


# read in the data
T = pd.read_csv("T.csv")

# collect the words from each document
# X is a term (columns) document (rows) matrix
X = pd.DataFrame()
for c in T.columns:
    vector = TfidfVectorizer()
    X2 = vector.fit_transform(T[c].tolist())
    X2 = pd.DataFrame(X2.toarray(), columns=vector.get_feature_names())
    X = pd.concat([X, X2], axis=1)

# standardize the data to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# train a PCA model
component = PCA(n_components=10, random_state=42)
component.fit(X)
X2 = component.transform(X)

# train a k-means model
cluster = KMeans(n_clusters=6, n_init=20, max_iter=300, tol=0.0001, random_state=42,
                 n_jobs=1)
cluster.fit(X2)

# compute clusters for all the data
Y = cluster.predict(X2)

# train random forest model
model = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=5,
                               max_features="sqrt", random_state=42, n_jobs=1)
model.fit(X, Y)

# collect word importance
words = pd.DataFrame({"Word": X.columns, 
                      "Importance": model.feature_importances_})
words = words.sort_values(by="Importance", ascending=False).reset_index(drop=True)

# export word importance
words.to_csv("words.csv", index=False)
