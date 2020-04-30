#!/usr/bin/env python3
""" Representation of a normal distribution"""


class Normal:
    """
    Class exponential distribution
    """
    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Class constructor
        :param data: is a list of the data to be used to estimate the
        distribution
        :param mean: is the mean of the distribution
        :param stddev: is the standard deviation of the distribution
        """
        self.mean = float(mean)
        self.stddev = float(stddev)
        if data is None:
            if stddev < 0:
                raise ValueError("stddev must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            res_std = []
            for i in range(len(data)):
                res_std.append((data[i] - self.mean) ** 2)
            self.stddev = (sum(res_std) / len(data)) ** (1 / 2)

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value
        :param x: is the x-value
        :return: the z-score of x
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score
        :param z: z is the z-score
        :return: the x-value of z
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        :param x: is the x-value
        :return: the PDF value for x
        """
        part1 = 1 / (self.stddev * ((2 * Normal.pi) ** 0.5))
        part2 = ((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        return part1 * Normal.e ** (-part2)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        :param x: is the x-value
        :return: the CDF value for x
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (Normal.pi ** 0.5)) * (
                    z - ((z ** 3) / 3) + ((z ** 5) / 10) - ((z ** 7) / 42) + (
                        (z ** 9) / 216))
        return 0.5 * (1 + erf)
