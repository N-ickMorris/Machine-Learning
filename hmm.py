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
train2, test2 = Y[0:size], Y[size:len(Y)]

# train and forecast the entire test set
history = [x for x in train]
model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
model.fit(history)
states = model.predict(test)
states_ = model.predict(train)

# collect states and actual values
states = pd.DataFrame({"Actual": test2.to_numpy().flatten(),
                       "Difference": test.flatten(),
                       "State": states.flatten(),
                       "Data": "Test"}).reset_index()
states_ = pd.DataFrame({"Actual": train2.to_numpy().flatten(),
                       "Difference": train.flatten(),
                       "State": states_.flatten(),
                       "Data": "Train"}).reset_index()
states.to_csv("hmm states - Test.csv", index=False)
states_.to_csv("hmm states - Train.csv", index=False)

# In[2]: Visualize the states

# plot the states on the train values
scatter_plot(data=states_,
             x="index", y="Actual",
             color="State", title="Hidden Markov Model - Train",
             legend=True, save=False)
scatter_plot(data=states_,
             x="index", y="Difference",
             color="State", title="Hidden Markov Model - Train",
             legend=True, save=False)
pairs_plot(data=states_, vars=["Actual", "Difference"],
           color="State", title="Hidden Markov Model - Train", save=False)

# plot the states on the test values
scatter_plot(data=states,
             x="index", y="Actual",
             color="State", title="Hidden Markov Model - Test",
             legend=True, save=False)
scatter_plot(data=states,
             x="index", y="Difference",
             color="State", title="Hidden Markov Model - Test",
             legend=True, save=False)
pairs_plot(data=states, vars=["Actual", "Difference"],
           color="State", title="Hidden Markov Model - Test", save=False)
