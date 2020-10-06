# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:33:43 2020

@author: Nick
"""


import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


# define training parameters
start_size = 100 # number of data points to start modeling with
horizon = 4 # number of data points to predict ahead
ar = 3 # number of autoregressive terms to model with
ma_window = 5 # number of data points to compute a moving average term

# convert series to supervised learning
def series_to_supervised(data, n_backward=1, n_forward=1, dropnan=False):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_backward, 0, -1):
		cols.append(df.shift(i))
		names += [(str(df.columns[j]) + '(t-%d)' % (i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_forward):
		cols.append(df.shift(-i))
		if i == 0:
			names += [str(df.columns[j]) + '(t)' for j in range(n_vars)]
		else:
			names += [(str(df.columns[j]) + '(t+%d)' % (i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# In[1]: Prepare the Data

# read in the data
data = pd.read_csv("data.csv")

# compute absolute percent deviation of Pressure from the Set Point
data["Deviation"] = (data["Pressure"] / data["Pressure_SP"] - 1).abs()

# convert the time stamp into a date time object
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

# convert Status into binary variables
status = pd.get_dummies(data[["Status"]].astype(str))
data = data.drop(columns="Status")
data = pd.concat([data, status], axis=1)

# define the inputs and outputs
inputs = ["Pressure", "Tempurature_001", "Tempurature_002", "Temperature_003",
          "Status_0", "Status_1", "Status_2", "Status_3", "Status_4"]
outputs = ["Deviation"]

# split the data into training
train = data[:start_size].copy()

# In[2]: Stream the Data

# define a function for streaming data
def incoming_data(df, idx):
    return df.iloc[[idx], :].copy()

predictions = pd.DataFrame()
actuals = pd.DataFrame()
time = []
for i in range(train.shape[0], data.shape[0] - horizon):
    # track time
    time.append(data.index[i])

    # split up inputs (X) and outputs (Y)
    X = train[inputs].copy()
    Y = train[[outputs[0]]].copy() # only single stream output allowed

    # add autoregressive terms to X
    X = pd.concat([X, series_to_supervised(Y, n_backward=ar, n_forward=1)], axis=1)

    # add a moving average term to X
    MA = Y.rolling(ma_window).mean()
    MA.columns = [c + " ma(" + str(ma_window) + ")" for c in MA.columns]
    X = pd.concat([X, MA], axis=1)

    # add the forecasting horizon to Y
    Y = series_to_supervised(Y, n_backward=0, n_forward=horizon + 1)
    Y = Y.drop(columns=Y.columns[0]) # time (t) is tracked in X

    # use the last row of train to predict the horizon
    X_new = X[-1:].copy()

    # get the horizon
    Y_new = data.iloc[i:(i + horizon)][outputs[0]].tolist()
    actuals = pd.concat([actuals, pd.DataFrame(Y_new).T], axis=0) 

    # drop rows with missing values
    df = pd.concat([X, Y], axis=1).dropna()
    X = df[X.columns]
    Y = df[Y.columns]

    # set up cross validation for time series
    tscv = TimeSeriesSplit(n_splits=3)
    folds = tscv.get_n_splits(X)

    # set up a machine learning pipeline
    pipeline = Pipeline([
        ('var', VarianceThreshold()),
        ('scale', MinMaxScaler()),
        ('model', MultiOutputRegressor(LassoCV(cv=folds, eps=1e-9, n_alphas=16,
                                               n_jobs=-1))),
    ])

    # train and forecast
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pipeline.fit(X, Y)
    forecast = pipeline.predict(X_new)
    predictions = pd.concat([predictions, pd.DataFrame(forecast)], axis=0) 

    # stream data
    train = pd.concat([train, incoming_data(data, i)], axis=0)

    # report on one step ahead prediction
    pred = forecast[0][0]
    true = Y_new[0]
    print('1-step ahead: predicted=%f, expected=%f' % (pred, true))

# In[2]: Evaluate the Model

# compute R2
R2 = r2_score(actuals, predictions, multioutput="raw_values")
R2 = pd.DataFrame({"R2": R2})
R2["Step Ahead"] = R2.index + 1
print(R2)

# pick a step ahead to evaluate
step_ahead = 1
df = pd.concat([actuals.iloc[:,step_ahead - 1],
                predictions.iloc[:,step_ahead - 1]], axis=1)
df.columns = ["Actual", "Predict"]
df["index"] = time

# plot the prediction series
fig = px.scatter(df, x="index", y="Predict")
fig.add_trace(go.Scatter(x=df["index"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig.update_layout(font=dict(size=16))
plot(fig, filename="Series Predictions.html")

# draw a parity plot
fig1 = px.scatter(df, x="Actual", y="Predict")
fig1.add_trace(go.Scatter(x=df["Actual"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"))
fig1.update_layout(font=dict(size=16))
plot(fig1, filename="Parity Plot.html")
