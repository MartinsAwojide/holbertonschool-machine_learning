#!/usr/bin/env python3
""" Representation of a exponential distribution"""


class Exponential:
    """
    Class exponential distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        :param data: a list of the data to be used to estimate the
        distribution
        :param lambtha: is the expected number of occurences in
        a given time frame
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given time period
        :param x: is the time period
        :return: the PDF value for x
        """
        if x < 0:
            return 0
        return self.lambtha * (Exponential.e ** (-x * self.lambtha))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        :param x: is the time period
        :return: the CDF value for x
        """
        result = 0
        if x < 0:
            return 0
        result = Exponential.e ** (-x * self.lambtha)
        print(result)
        return 1 - result
