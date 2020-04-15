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
        if i in axes.keys():
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None, None, None))
    return matrix[tuple(slices)]
