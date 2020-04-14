#!/usr/bin/env python3


def matrix_transpose(matrix):
    """A funtion that transpose a 2d matrix

    Arguments:
        matrix {[list]} -- lists of ints/floats

    Returns:
        [type] -- New list
    """
    try:
        return [[matrix[j][i] for j in range(len(matrix))]
                for i in range(len(matrix[0]))]
    except (TypeError, IndexError):
        return [[matrix[i]] for i in range(len(matrix))]
