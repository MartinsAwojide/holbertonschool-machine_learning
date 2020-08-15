#!/usr/bin/env python3
"""Markov models"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
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

    Returns: P, F, or None, None on failure
    - P is the likelihood of the observations given the model
    - F is a numpy.ndarray of shape (N, T) containing the forward
    path probabilities
    """
    if type(Observation) is not np.ndarray or len(Observation.shape) != 1:
        return None, None
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    if not (np.sum(Emission, axis=1) == 1.0).all():
        return None, None
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    if not (np.sum(Transition, axis=1) == 1.0).all():
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[0] != Emission.shape[0]:
        return None, None
    if Transition.shape[0] != Transition.shape[1]:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    if not (np.sum(Initial, axis=1) == 1.0).all():
        return None, None
    N, M = Emission.shape
    T = Observation.shape[0]
    alpha = np.zeros((N, T))
    # dont have T emissions so, work through columns instead of rows
    alpha[:, 0] = Initial.T * Emission[:, Observation[0]]

    for col in range(1, T):
        for row in range(N):
            aux = np.dot(Transition[:, row], Emission[row, Observation[col]])
            alpha[row, col] = np.dot(alpha[:, col - 1], aux)

    P = np.sum(alpha[:, -1])
    return P, alpha
