#!/usr/bin/env python3
'''adds two multidemsional matrices'''


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


def add_matrices(mat1, mat2):
    """adds two multidemsional matrices

    Arguments:
        mat1 {[ndarray]}
        mat2 {[ndarray]}

    Returns:
        [ndarray]
    """
    if matrix_shape(mat1) == matrix_shape(mat2):
        if type(mat1[0]) is not list:
            result = [mat1[i] + mat2[i] for i in range(len(mat1))]
            return result
        else:
            result = []
            for i in range(len(mat1)):
                aux = add_matrices(mat1[i], mat2[i])
                result.append(aux)
            return result
    else:
        return None
