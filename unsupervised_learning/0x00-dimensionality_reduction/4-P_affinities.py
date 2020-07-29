#!/usr/bin/env python3
"""Calculates the symmetric P affinities of a data set"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set
    Args:
        X: is a numpy.ndarray of shape (n, d) containing the dataset
        to be transformed by t-SNE
        - n is the number of data points
        - d is the number of dimensions in each point
        tol: is the maximum tolerance allowed (inclusive) for
        the difference in Shannon entropy from
        perplexity: is the perplexity that all Gaussian distributions
        should have

    Returns:
    P, a numpy.ndarray of shape (n, n) containing the symmetric P affinities
    """
    n, _ = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        copy = D[i]
        copy = np.delete(copy, i, axis=0)
        Hi, Pi = HP(copy, betas[i])
        betamin = None
        betamax = None
        Hdiff = Hi - H

        while np.abs(Hdiff) > tol:
            if Hdiff > 0:
                betamin = betas[i, 0]
                if betamax is None:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + betamax) / 2
            else:
                betamax = betas[i, 0]
                if betamin is None:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + betamin) / 2
            # Recompute the values
            Hi, Pi = HP(copy, betas[i])
            Hdiff = Hi - H
        # Set the final row of P, reinserting the missing spot as 0
        aux = np.insert(Pi, i, 0)
        P[i] = aux
    # The symmetrized conditional probabilities
    P = (P.T + P) / (2*n)
    return P
