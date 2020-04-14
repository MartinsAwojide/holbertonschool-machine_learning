#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    """A function that performs matrix multiplication

    Arguments:
        mat1 {[list} -- 2D matrices containing ints/floats
        mat2 {[list]} -- 2D matrices containing ints/floats

    Returns:
        [list] -- new matrix withe the result of multiplication
    """
    result = []
    if len(mat1[0]) == len(mat2):
        for i in range(len(mat1)):
            aux = []
            for k in range(len(mat2[0])):
                num = 0
                for j in range(len(mat1[i])):
                    num += mat1[i][j] * mat2[j][k]
                aux.append(num)
            result.append(aux)
        return result
    return None
