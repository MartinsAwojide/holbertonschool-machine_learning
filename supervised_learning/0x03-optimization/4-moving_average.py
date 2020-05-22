#!/usr/bin/env python3
"""mini batch"""
import numpy as np


def moving_average(data, beta):
    """
    calculates the weighted moving average of a data set
    :param data: is the list of data to calculate the moving average of
    :param beta: is the weight used for the moving average
    :return: a list containing the moving averages of data
    """
    vt = 0
    moving_average = []
    for i in range(len(data)):
        vt = beta * vt + (1-beta) * data[i]  # beta*vt, vt is vt-1
        den = 1 - beta ** (i+1)  # 1-beta^t bias correction.+1 as i starts in 0
        moving_average.append(vt / den)

    return moving_average
