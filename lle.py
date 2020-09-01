# -*- coding: utf-8 -*-
"""
Reduces the columns of a data set with a Locally Linear Embedding model

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.manifold import LocallyLinearEmbedding
from plots import scatter_plot, corr_plot


# read in the data
X = pd.read_csv("X clean.csv")

# standardize the data to take on values between 0 and 1
X = (X - X.min()) / (X.max() - X.min())

# separate the data into training and testing
np.random.seed(1)
test_idx = np.random.choice(a=X.index.values, size=int(X.shape[0] / 5), replace=False)
train_idx = np.array(list(set(X.index.values) - set(test_idx)))

# train a LocallyLinearEmbedding model
n_comp = 1 # number of components
component = LocallyLinearEmbedding(n_components=n_comp, n_neighbors=20, n_jobs=1, 
                                   random_state=42)
component.fit(X.iloc[train_idx, :])

# compute components for all the data, add cluster labels and train/test labels
components = pd.DataFrame(component.transform(X), 
                          columns=["LC" + str(i + 1) for i in range(n_comp)])
components["Data"] = "Train"
for j in test_idx:
    components.loc[j, "Data"] = "Test"
# components.to_csv("lle.csv", index=False)

# combine the data and components
data = pd.concat([X, components], axis=1)

# plot the variables vs. components
comp_ = 0 # the column number of a component to plot
for c in X.columns:
    scatter_plot(data=data,
                 x=c, y=components.columns[comp_],
                 color=None, title="Locally Linear Embedding",
                 legend=True, save=False)

# plot correlations
corr_plot(data.drop(columns="Data"))
