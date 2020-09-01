# -*- coding: utf-8 -*-
"""
Cleans up a data set to have no outliers

@author: Nick
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor


# read in the data
X = pd.read_csv("X clean.csv")
Y = pd.read_csv("Y clean.csv")

# train a model to detect outliers
data = pd.concat([Y, X], axis=1)
model = LocalOutlierFactor(n_neighbors=20, leaf_size=30, novelty=False, n_jobs=1)
model.fit(data)

# remove 2% of the data
percent = 0.02
cutoff = np.quantile(model.negative_outlier_factor_, percent)
good_idx = np.where(model.negative_outlier_factor_ > cutoff)[0]
X = X.iloc[good_idx, :].reset_index(drop=True)
Y = Y.iloc[good_idx, :].reset_index(drop=True)

# export the data
X.to_csv("X clean.csv", index=False)
Y.to_csv("Y clean.csv", index=False)
