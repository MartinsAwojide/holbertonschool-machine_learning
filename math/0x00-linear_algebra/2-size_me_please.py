#!/usr/bin/env python3


def matrix_shape(matrix):
    """A funtion that calculates the shape of a matrix

    Arguments:
        matrix {[list]} -- lists of ints/floats

    Returns:
        [list] -- New list
    """
    shape = []
    try:
        shape.append(len(matrix))
        try:
            shape.append(len(matrix[0]))
        except (TypeError, IndexError):
            return shape
        try:
            shape.append(len(matrix[0][0]))
        except (TypeError, IndexError):
            return shape
        return shape

    except (TypeError, IndexError):
        return shape
