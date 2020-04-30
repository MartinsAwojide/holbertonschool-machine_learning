#!/usr/bin/env python3
""" Representation of a poisson distribution"""


class Poisson:
    """
    Class poisson distribution
    """
    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Class constructor
        :param data: a list of the data to be used to estimate the distribution
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
                self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        :param k: is the number of “successes”
                  If k is not an integer, convert it to an integer
                  If k is out of range, return 0
        :return: the PMF value for k
        """
        factorial = 1
        if k < 0:
            return 0
        if k is not int:
            int(k)
        for i in range(1, k + 1):
            factorial *= i
        pmf = Poisson.e ** (-self.lambtha) * self.lambtha ** k / factorial
        return pmf

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        :param k: is the number of “successes”
                  If k is not an integer, convert it to an integer
                  If k is out of range, return 0
        :return: the CDF value for k
        """
        result = []
        for i in range(k + 1):
            result.append(self.pmf(i))
        return sum(result)
