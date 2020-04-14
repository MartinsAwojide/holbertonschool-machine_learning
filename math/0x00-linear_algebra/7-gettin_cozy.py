#!/usr/bin/env python3


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis

    Arguments:
        mat1 {[list]} -- 2D matrices containing ints/floats
        mat2 {[list]} -- 2D matrices containing ints/floats

    Keyword Arguments:
        axis {int} -- Specific axis (default: {0})

    Returns:
        [list] -- A new matrix
    """
    concat = []
    if len(mat1[0]) == len(mat2[0]) and axis == 0:
        concat = [row[:] for row in mat1] \
            + [row.copy() for row in mat2]
        return concat
    if len(mat1) == len(mat2) and axis == 1:
        new = [mat1[i] + mat2[i] for i in range(len(mat1))]
        return new
    else:
        return None
