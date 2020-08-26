# -*- coding: utf-8 -*-
"""
Creates plots for analyzing model performance

@author: Nick
"""


import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def matrix_plot(matrix, title=" ", save=False):
    # set up labels for the plot
    group_names = ["True Neg","False Pos","False Neg","True Pos"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         matrix.flatten()/np.sum(matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    # plot the predictions
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=labels, fmt="", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predict")
    ax.set_ylabel("Actual")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def parity_plot(predict, actual, title=" ", alpha=2/3, save=False):
    # plot the predictions
    fig, ax = plt.subplots()
    sns.scatterplot(actual, predict, color="blue", alpha=alpha, ax=ax)
    sns.lineplot(actual, actual, color="red", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Predict")
    ax.set_xlabel("Actual")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def pairs_plot(data, vars, color, title=" ", save=False):
    p = sns.pairplot(data, vars=vars, hue=color)
    p.fig.suptitle(title, y=1.08)
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def series_plot(predict, actual, title=" ", alpha=2/3, save=False):
    # plot the predictions
    fig, ax = plt.subplots()
    idx = [i for i in range(len(predict))]
    sns.scatterplot(idx, predict, color="blue", alpha=alpha, ax=ax)
    sns.lineplot(idx, actual, color="red", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()
