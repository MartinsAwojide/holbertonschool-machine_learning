#!/usr/bin/env python3
"""Markov models"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    Args:
        Observation: is a numpy.ndarray of shape (T,) that contains the index
        of the observation
        - T is the number of observations
        Emission: is a numpy.ndarray of shape (N, M) containing the emission
        probability of a specific observation given a hidden state
        - Emission[i, j] is the probability of observing j given
        the hidden state i
        - N is the number of hidden states
        - M is the number of all possible observations
        Transition: is a 2D numpy.ndarray of shape (N, N) containing
        the transition probabilities
        - Transition[i, j] is the probability of transitioning from the
        hidden state i to j
        Initial: numpy.ndarray of shape (N, 1) containing the probability
        of starting in a particular hidden state

    Returns: P, beta, or None, None on failure
    - P is the likelihood of the observations given the model
    - beta is a numpy.ndarray of shape (N, T) containing the backward
    path probabilities
    - beta[i, j] is the probability of generating the future observations
    from hidden state i at time j
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None

    if not (np.sum(Emission, axis=1) == 1.0).all():
        return None, None
    if not (np.sum(Transition, axis=1) == 1.0).all():
        return None, None
    if not (np.sum(Initial, axis=0) == 1.0).all():
        return None, None

    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[0] != Emission.shape[0]:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    N, M = Emission.shape
    T = Observation.shape[0]
    beta = np.zeros((N, T))
    # dont have T emissions so, work through columns instead of rows
    beta[:, T - 1] = np.ones(N)

    for col in range(T - 2, -1, -1):
        for row in range(N):
            aux = Emission[:, Observation[col + 1]] * Transition[row, :]
            beta[row, col] = np.dot(beta[:, col + 1], aux)

    # respect to column
    P = np.sum(Initial.T * Emission[:, Observation[0]] * beta[:, 0])
    return P, beta
