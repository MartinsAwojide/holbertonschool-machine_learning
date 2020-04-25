#!/usr/bin/env python3
"""calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """calculates the derivative of a polynomial

    Arguments:
        :param poly: the index of the list represents the power
        of x that the coefficient belongs to
        Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
        If poly is not valid, return None
        If the derivative is 0, return [0]

    Returns:
        new list of coefficients representing the derivative of the polynomial
    """
    if len(poly) == 0 or type(poly) is not list:
        return None
    if len(poly) == 1:
        return [0]
    derivative = []
    poly.pop(0)
    i = 0
    while i < len(poly):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        derivative.append(poly[i] * (i + 1))
        i += 1
    return derivative
