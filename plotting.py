# -*- coding: utf-8 -*-
"""
Creates plots for analyzing data

@author: Nick
"""


import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='ward')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

df = pd.read_csv("X clean.csv").iloc[:,:10]

def cormap(df, title=None, font_size=None):
    corr = cluster_corr(df.corr())
    fig = px.imshow(corr, x=corr.index.tolist(), y=corr.columns.tolist(), 
                    title=title, labels=dict(color="Correlation"))
    fig.update_layout(font=dict(size=font_size))
    plot(fig)

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
