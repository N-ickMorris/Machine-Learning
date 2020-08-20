# -*- coding: utf-8 -*-
"""
Creates Isomap Embeddings
Creates Spectral Embeddings

@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.manifold import Isomap, SpectralEmbedding


# read in the data
X = pd.read_csv("X clean.csv")

# create isomap embeddings
num = 6
isomap = Isomap(n_neighbors=20, n_components=num, n_jobs=1)
isoX = pd.DataFrame(isomap.fit_transform(X), 
                    columns=["I" + str(n + 1) for n in range(num)])

# create spectral embeddings
num = 6
spectral = SpectralEmbedding(n_neighbors=20, n_components=num, random_state=42, n_jobs=1)
speX = pd.DataFrame(spectral.fit_transform(X), 
                    columns=["S" + str(n + 1) for n in range(num)])

X = pd.concat([X, isoX, speX], axis=1)

# export the data
X.to_csv("X clean.csv", index=False)
