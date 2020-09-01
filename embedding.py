# -*- coding: utf-8 -*-
"""
Creates Isomap Embeddings

@author: Nick
"""


import pandas as pd
from sklearn.manifold import Isomap


# read in the data
X_copy = pd.read_csv("X clean.csv")

# standardize the data to take on values between 0 and 1
X = ((X_copy - X_copy.min()) / (X_copy.max() - X_copy.min())).copy()

# create isomap embeddings
num = 6
isomap = Isomap(n_neighbors=5, n_components=num, n_jobs=1)
isoX = pd.DataFrame(isomap.fit_transform(X), 
                    columns=["I" + str(n + 1) for n in range(num)])

X = pd.concat([X_copy, isoX], axis=1)

# export the data
X.to_csv("X clean.csv", index=False)
