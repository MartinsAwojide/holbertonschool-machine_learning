#!/usr/bin/env python3

import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis

    Arguments:
        mat1 {[ndarray]}
        mat2 {[ndarray]}

    Keyword Arguments:
        axis {ndarray} -- [axis] (default: {0})

    Returns:
        [ndarray]
    """
    return np.concatenate((mat1, mat2), axis=axis)
