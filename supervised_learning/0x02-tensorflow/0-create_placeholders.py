#!/usr/bin/env python3
"""Placeholders"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    return two placeholders
    :param nx: the number of feature columns in our data
    :param classes: the number of classes in our classifier
    :return: placeholders named x and y, respectively
    """
    x = tf.placeholder("float", [None, nx])
    y = tf.placeholder("float", [None, classes])
    return x, y
