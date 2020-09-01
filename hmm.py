# -*- coding: utf-8 -*-
"""
Trains and tests a Hidden Markov Model on data

@author: Nick
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from plots import scatter_plot, pairs_plot

# In[1]: Train the model

# load dataset
Y = pd.read_csv('Y time.csv').iloc[:,[0]]

# split up the data into training and testing
X = Y.values
size = int(len(X) * 0.8)
train, test = X[0:size], X[size:len(X)]

# train and forecast one step ahead for the entire test set
history = [x for x in train]
states = list()
for t in range(len(test)):
	model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=100)
	model.fit(history)
	obs = [test[t]]
	output = model.predict(obs)
	yhat = output[0]
	history.append(obs[0])
	states.append(yhat)
	print('state=%f, observed=%f' % (yhat, obs[0][0]))

# collect states and actual values
states = pd.DataFrame({"Actual": test.flatten(), 
                       "State": np.array(states).flatten()}).reset_index()
states.to_csv("hmm states.csv", index=False)

# In[2]: Visualize the states

# plot the states on the predicted values
scatter_plot(data=states,
             x="index", y="Actual",
             color="State", title="Hidden Markov Model",
             legend=True, save=False)

pairs_plot(data=states, vars=["index", "Actual"], 
           color="State", title="Hidden Markov Model", save=False)
