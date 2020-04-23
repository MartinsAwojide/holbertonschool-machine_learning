#!/usr/bin/env python3
'''calculates the integral of a polynomial'''


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial

    Arguments:
        poly is a list of coefficients representing a polynomial
        the index of the list represents the
        power of x that the coefficient belongs to
        Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]

    Keyword Arguments:
        C is an integer representing the integration constant
        If a coefficient is a whole number,
        it should be represented as an integer

    Returns:
        new list of coefficients representing the integral of the polynomial
    """
    if len(poly) == 0:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    integral = [C]
    for i in range(len(poly)):
        if type(poly[i]) is not int and type(poly[i]) is not float:
            return None
        if i == 0:
            integral.append(poly[i])
        else:
            aux = poly[i]/(i+1)
            if aux % 1 == 0:
                integral.append(int(aux))
            else:
                integral.append(aux)
    return integral
