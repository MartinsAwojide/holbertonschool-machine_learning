#!/usr/bin/env python3
"""Markov models"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    forward hidden Markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]
    alpha = np.zeros((N, T))
    aux = (Initial.T * Emission[:, Observation[0]])
    alpha[:, 0] = aux.reshape(-1)
    for t in range(1, T):
        for n in range(N):
            prev = alpha[:, t - 1]
            trans = Transition[:, n]
            em = Emission[n, Observation[t]]
            alpha[n, t] = np.sum(prev * trans * em)

    prop = np.sum(alpha[:, -1])
    return (prop, alpha)


def backward(Observation, Emission, Transition, Initial):
    """
    backward hidden Markov model
    """
    N, M = Emission.shape
    T = Observation.shape[0]

    # we should initialize the last prob as 1
    beta = np.zeros((N, T))
    beta[:, T - 1] = 1

    # this should be start in T-2 and should stop in 0
    # that is why the range has this form and step
    for t in range(T - 2, -1, -1):
        for n in range(N):
            trans = Transition[n]
            em = Emission[:, Observation[t + 1]]
            post = beta[:, t + 1]
            first = post * em
            beta[n, t] = np.dot(first.T, trans)

    prob = np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0])
    return (prob, beta)


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a hidden markov model
    Args:
        Observations: is a numpy.ndarray of shape (T,) that contains
        the index of the observation
        - T is the number of observations
        Transition: is a numpy.ndarray of shape (M, M) that contains
        the initialized transition probabilities
        - M is the number of hidden states
        Emission: is a numpy.ndarray of shape (M, N) that contains
        the initialized emission probabilities
        - N is the number of output states
        Initial: is a numpy.ndarray of shape (M, 1) that contains
        the initialized starting probabilities
        iterations: is the number of times expectation-maximization
        should be performed

    Returns: the converged Transition, Emission, or None, None on failure
    """
    if type(Observations) is not np.ndarray or len(Observations.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[0] != Emission.shape[0]:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None

    N, M = Emission.shape
    T = Observations.shape[0]
    a = Transition.copy()
    b = Emission.copy()
    for n in range(iterations):
        _, alpha = forward(Observations, b, a, Initial.reshape((-1, 1)))
        _, beta = backward(Observations, b, a, Initial.reshape((-1, 1)))

        xi = np.zeros((N, N, T - 1))
        for col in range(T - 1):
            denominator = np.dot(np.dot(alpha[:, col].T, a) *
                                 b[:, Observations[col + 1]].T,
                                 beta[:, col + 1])
            for row in range(N):
                numerator = alpha[row, col] * a[row, :] * \
                            b[:, Observations[col + 1]].T * beta[:, col + 1].T
                xi[row, :, col] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack(
            (gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        denominator = np.sum(gamma, axis=1)
        for k in range(M):
            b[:, k] = np.sum(gamma[:, Observations == k], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b
