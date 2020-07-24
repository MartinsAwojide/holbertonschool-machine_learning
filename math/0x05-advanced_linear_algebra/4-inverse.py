#!/usr/bin/env python3
"""Calculates the inverse of a matrix"""


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    Args:
        matrix: is a list of lists whose determinant should be calculated

    Returns: The determinant of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if matrix == [[]]:
        return 1
    for i in range(len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(matrix[i]):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    cof = 1
    det = 0
    for i in range(len(matrix)):
        copy = []
        for ele in matrix:
            copy.append(ele.copy())

        new_matrix = copy[1:]
        for delete in new_matrix:
            del delete[i]

        det += matrix[0][i] * determinant(new_matrix) * cof
        cof = cof * -1

    return det


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix
    Args:
        matrix: s a list of lists whose minor matrix should be calculated

    Returns: the cofactor matrix of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    for i in range(len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(matrix) != len(matrix[i]) or len(matrix[i]) == 0:
            raise ValueError("matrix must be a non-empty square matrix")

    new_matrix = []
    for i in range(len(matrix)):
        aux = []
        for j in range(len(matrix)):
            copy = []
            for ele in matrix:
                copy.append(ele.copy())
            del copy[i]
            for delete in copy:
                del delete[j]

            if (i+j) % 2 != 0:
                cof = -1
            else:
                cof = 1
            aux.append(determinant(copy) * cof)
        new_matrix.append(aux)
    return new_matrix


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    Args:
        matrix: is a list of lists whose adjugate matrix should be calculated

    Returns: the adjugate matrix of matrix
    """
    cof = cofactor(matrix)
    adjugate = []

    for i in range(len(cof)):
        copy = []
        for j in range(len(cof)):
            copy.append(cof[j][i])
        adjugate.append(copy)

    return adjugate


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    Args:
        matrix: is a list of lists whose inverse should be calculated

    Returns: the inverse of matrix, or None if matrix is singular
    """
    adj = adjugate(matrix)
    det = determinant(matrix)
    if det == 0:
        return None
    inv = []
    for i in range(len(adj)):
        aux = []
        for j in range(len(adj)):
            aux.append(adj[i][j] / det)
        inv.append(aux)
    return inv
