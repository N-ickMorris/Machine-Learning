# -*- coding: utf-8 -*-
"""
Trains and tests a Hidden Markov Model on data

@author: Nick
"""

import pandas as pd
from hmmlearn import hmm
from plots import scatter_plot, pairs_plot

# In[1]: Train the model

# load dataset
Y = pd.read_csv('Y time.csv').iloc[:,[0]]

# difference the data by 1 period
Y_diff = Y.diff().fillna(value=0)

# split up the data into training and testing
X = Y_diff.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]
test2 = Y[size:len(Y)]

# train and forecast the entire test set
history = [x for x in train]
model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
model.fit(history)
states = model.predict(test)

# collect states and actual values
states = pd.DataFrame({"Actual": test2.to_numpy().flatten(),
                       "Difference": test.flatten(),
                       "State": states.flatten()}).reset_index()
states.to_csv("hmm states.csv", index=False)

# In[2]: Visualize the states

# plot the states on the actual values
scatter_plot(data=states,
             x="index", y="Actual",
             color="State", title="Hidden Markov Model",
             legend=True, save=False)
scatter_plot(data=states,
             x="index", y="Difference",
             color="State", title="Hidden Markov Model",
             legend=True, save=False)
pairs_plot(data=states, vars=["Actual", "Difference"],
           color="State", title="Hidden Markov Model", save=False)
