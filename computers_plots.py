# -*- coding: utf-8 -*-
"""
https://plotly.com/python/plotly-express/

@author: Nick
"""

import pandas as pd
import plotly.express as px
from plotly.offline import plot

df = pd.read_csv("Computers.csv")

# define the target feature and other meaningful features
targets = ["price"]
features = ["speed", "hd", "ram", "screen", "ads", "trend"]
colors = ["cd", "multi", "premium"]

# plot a matrix of scatter plots to see the whole data set
fig = px.scatter_matrix(df, 
                        dimensions=targets + features,
                        color=colors[2],
                        opacity=0.7)
fig.update_traces(diagonal_visible=False)
plot(fig)

# these 3 variables have the strongest separation for the target
group = ["price", "ram", "hd"]

# plot the group of 3 variables across categories
fig = px.scatter_3d(df, x=group[0], y=group[1], z=group[2],
                    color=colors[2], opacity=0.7)
plot(fig)

# plot two variables across categories
fig = px.density_contour(df, x=group[0], y=group[1], marginal_x="histogram", 
                         marginal_y="box", color=colors[2])
plot(fig)

# plot a singl variable across categories
fig = px.strip(df, y=group[1], color=colors[2])
plot(fig)
