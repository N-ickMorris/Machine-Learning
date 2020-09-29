# -*- coding: utf-8 -*-
"""
Creates plots for analyzing data

@author: Nick
"""


import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot

def scatter(df, x, y, color=None, title=None, font_size=None):
    fig = px.scatter(df, x=x, y=y, color=color, title=title)
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def line(df, x, y, color=None, title=None, font_size=None):
    fig = px.line(df, x=x, y=y, color=color, title=title)
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def parity(df, predict, actual, color=None, title=None, font_size=None):
    fig = px.scatter(df, x=actual, y=predict, color=color, title=title)
    fig.add_trace(go.Scatter(x=df[actual], y=df[actual], mode="lines", showlegend=False, name="Actual"))
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def series(df, predict, actual, color=None, title=None, font_size=None):
    df = df.reset_index()
    fig = px.scatter(df, x="index", y=predict, color=color, title=title)
    fig.add_trace(go.Scatter(x=df["index"], y=df[actual], mode="lines", showlegend=False, name="Actual"))
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def histogram(df, x, color=None, title=None, font_size=None):
    fig = px.histogram(df, x=x, color=color, title=title, marginal="box")
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

def pairs(df, color=None, title=None, font_size=None):
    fig = px.scatter_matrix(df, color=color, title=title)
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(font=dict(size=font_size))
    plot(fig)
