# coding: utf-8
# !/usr/bin/env python3
# --------------------------------------------------
#
# 
# written by: Apoi
# version: 
#
# --------------------------------------------------
#
# PROJECT:
#
#
# FILE PURPOSE:
#
#
# FILE ISSUE:
#
#
# --------------------------------------------------

# --------------------------------------------------
# library importing
# --------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------
# python file importing
# --------------------------------------------------

# --------------------------------------------------
# class
# --------------------------------------------------


def visualize_classifier(classifier, vx, y, title=''):
    min_x, max_x = vx[:, 0].min() - 1.0, vx[:, 0].max() + 1.0
    min_y, max_y = vx[:, 1].min() - 1.0, vx[:, 1].max() + 1.0

    mesh_step_size = 0.01

    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                 np.arange(min_y, max_y, mesh_step_size))

    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    plt.figure()
    plt.title(title)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
    plt.scatter(vx[:, 0], vx[:, 1], c=y, s=75, edgecolors='black', linewidths=1,
                cmap=plt.cm.Paired)

    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    plt.xticks((np.arange(int(min_x), int(max_x), 1.0)))
    plt.yticks((np.arange(int(min_y), int(max_y), 1.0)))

    plt.show()
