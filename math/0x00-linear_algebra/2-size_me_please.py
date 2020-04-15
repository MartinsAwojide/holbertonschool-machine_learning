#!/usr/bin/env python3
'''calculates the shape of a matrix'''


def matrix_shape(matrix):
    """A funtion that calculates the shape of a matrix

    Arguments:
        matrix {[list]} -- lists of ints/floats

    Returns:
        [list] -- New list
    """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])
