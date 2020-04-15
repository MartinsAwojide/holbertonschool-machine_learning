#!/usr/bin/env python3


def np_slice(matrix, axes={}):
    """Function that slices a matrix along a specific axes

    Arguments:
        matrix {[ndarray]}

    Keyword Arguments:
        axes {dict} -- is a dictionary where the key is an
        axis to slice along and the value is a tuple representing
        the slice to make along that axis

    Returns:
        [ndarray]
    """
    slices = []
    for i in range(len(matrix.shape)):
        for keys, values in axes.items():
            if i == 0 and keys != 0:
                slices.append(slice(None, None, None))
            if keys == i:
                slices.append(slice(*values))
    return matrix[tuple(slices)]
