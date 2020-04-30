#!/usr/bin/env python3
""" Representation of a binomial distribution"""


class Binomial:
    """
    Class binomial distribution
    """
    def __init__(self, data=None, n=1, p=0.5):
        """
        Class constructor
        :param data: a list of the data to be used to estimate the distribution
        :param n: is the number of Bernoulli trials
        :param p: is the probability of a “success”
        """
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n < 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = []
            for i in range(len(data)):
                var.append((data[i] - mean) ** 2)
            variance = sum(var) / len(data)

            self.p = 1 - (variance / mean)
            self.n = int(round(mean/self.p))
            self.p = mean / self.n

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        :param k: is the number of “successes”
        :return: the PMF value for k
                 If k is not an integer, convert it to an integer
                 If k is out of range, return 0
        """
        if k is not int:
            int(k)
        if k < 0 or k > self.n:
            return 0
        else:
            factorial_n = 1
            factorial_k = 1
            factorial_c = 1
            for i in range(1, k + 1):
                factorial_k *= i
            for i in range(1, self.n + 1):
                factorial_n *= i
            for i in range(1, (self.n - k) + 1):
                factorial_c *= i
            combinatorial = factorial_n / (factorial_k * factorial_c)
            return combinatorial * ((self.p**k) * ((1 - self.p)**(self.n - k)))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        :param k: is the number of “successes”
        :return: the CDF value for k
                 If k is not an integer, convert it to an integer
                 If k is out of range, return 0
        """
        result = []
        for i in range(k + 1):
            result.append(self.pmf(i))
        return sum(result)
