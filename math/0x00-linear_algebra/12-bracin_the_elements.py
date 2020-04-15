#!/usr/bin/env python3
'''operations'''


def np_elementwise(mat1, mat2):
    """Function that performs element-wise addition, subtraction,
     multiplication, and division

    Arguments:
        mat1 {[ndarray]}
        mat2 {[ndarray]}

    Returns:
        [ndarray]
    """
    return mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2
