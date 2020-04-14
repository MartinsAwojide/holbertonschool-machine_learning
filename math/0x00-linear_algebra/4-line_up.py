#!/usr/bin/env python3


def add_arrays(arr1, arr2):
    """A function that  adds two arrays element-wise

    Arguments:
        arr1 {[list]} --  lists of ints/floats
        arr2 {[list]} --  lists of ints/floats

    Returns:
        [list] -- New list
    """
    result = []

    if (len(arr1) == len(arr2)):
        for i in range(len(arr1)):
            result.append(arr1[i] + arr2[i])
        return result
    else:
        return None
