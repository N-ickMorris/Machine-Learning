# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 20:45:48 2020

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
    sns.scatterplot(predict, actual, color="blue", alpha=alpha, ax=ax)
    sns.lineplot(actual, actual, color="red", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predict")
    ax.set_ylabel("Actual")
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()

def pairs_plot(data, vars, color, title, save=False):
    fig, ax = plt.subplots()
    sns.pairplot(data, vars=vars, hue=color)
    if save:
        title = re.sub("[^A-Za-z0-9]+", "", title)
        plt.savefig(title + ".png")
    else:
        plt.show()