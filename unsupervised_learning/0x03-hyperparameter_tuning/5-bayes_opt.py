#!/usr/bin/env python3
"""Noiseless 1D Gaussian process"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor
        Args:
            f: is the black-box function to be optimized
            X_init: is a numpy.ndarray of shape (t, 1) representing the
            inputs already sampled with the black-box function
            Y_init: is a numpy.ndarray of shape (t, 1) representing
            the outputs of the black-box function for each input in X_init
            bounds: is a tuple of (min, max) representing the bounds of
            the space in which to look for the optimal point
            ac_samples: is the number of samples that should be analyzed
            during acquisition
            l: is the length parameter for the kernel
            sigma_f: is the standard deviation given to the output
            of the black-box function
            xsi: is the exploration-exploitation factor for acquisition
            minimize: is a bool determining whether optimization should
            be performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min, max = bounds
        self.X_s = np.linspace(min, max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
        Returns: X_next, EI
        - X_next is a numpy.ndarray of shape (1,) representing the next
        best sample point
        - EI is a numpy.ndarray of shape (ac_samples,) containing
        the expected improvement of each potential sample

        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is False:
            mu_sample_opt = np.amax(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi
        else:
            mu_sample_opt = np.amin(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi

        Z = np.zeros(sigma.shape)
        for i in range(len(sigma)):
            if sigma[i] != 0:
                Z[i] = imp[i] / sigma[i]
            else:
                Z[i] = 0
        ei = np.zeros(sigma.shape)
        for i in range(len(sigma)):
            if sigma[i] > 0:
                ei[i] = imp[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
            else:
                ei[i] = 0

        X_next = self.X_s[np.argmax(ei)]

        return X_next, ei

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function
        Args:
            iterations: is the maximum number of iterations to perform

        Returns: X_opt, Y_opt
        - X_opt is a numpy.ndarray of shape (1,) representing the optimal point
        - Y_opt is a numpy.ndarray of shape (1,) representing the optimal
        function value
        """
        positions = []
        for i in range(iterations):
            # Find the next sampling point xt by optimizing the acquisition
            # function over the GP: xt = argmaxx μ(x | D1:t−1)
            x_new, _ = self.acquisition()
            # If the next proposed point is one that has already been sampled,
            # optimization should be stopped early
            if x_new in positions:
                break
            y_new = self.f(x_new)
            # Add the sample to previous samples
            # D1: t = {D1: t−1, (xt, yt)} and update the GP
            self.gp.update(x_new, y_new)
            positions.append(x_new)
        if self.minimize is True:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmax(self.gp.Y)
        x_new = self.gp.X[index]
        y_new = self.gp.Y[index]
        return x_new, y_new
