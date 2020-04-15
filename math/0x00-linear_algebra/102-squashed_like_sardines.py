#!/usr/bin/env python3
'''concatenates two multidimesional matrices'''


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


def cat_matrices(mat1, mat2, axis=0):
    """concatenates two multidimensional matrices along a specific axis

    Arguments:
        mat1 {[ndarray]}
        mat2 {[ndarray]}

    Keyword Arguments:
        axis {int} -- axis where to concatenate (recursion exit condition: {0})

    Returns:
        [ndarray]
    """
    if len(matrix_shape(mat1)) == len(matrix_shape(mat2)):

        result = []

        if axis == 0:
            result += [i for i in mat1]
            result += [i for i in mat2]
            return result
        else:
            for i in range(len(mat1)):
                aux = cat_matrices(mat1[i], mat2[i], axis-1)
                result.append(aux)
            return result
    else:
        None
