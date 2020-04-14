#!/usr/bin/env python3


def add_matrices2D(mat1, mat2):
    """Function that adds two matrices element-wise

    Arguments:
        mat1 {[list]} -- 2D matrices containing ints/floats
        mat2 {[list]} -- 2D matrices containing ints/floats

    Returns:
        [list] -- A new matrix
    """
    result = []
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        for i in range(len(mat1)):
            result.append([])
            for j in range(len(mat1[i])):
                result[i].append(mat1[i][j] + mat2[i][j])

        return result
    else:
        return None
