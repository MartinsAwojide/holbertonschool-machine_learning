#!/usr/bin/env python3
'''summatory calculation'''


def summation_i_squared(n):
    """summation of i^2

    Arguments:
        n is the stopping condition

    Returns:
        the integer value of the sum
    """
    if type(n) is not int or n < 1 or n is None:
        return None
    else:
        return int(n * (n + 1) * (2 * n + 1) / 6)
