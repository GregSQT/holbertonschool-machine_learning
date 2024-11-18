#!/usr/bin/env python3
"""
Advanced Linear Algebra Determinant
"""


def minor_m(m, row, col):
    """Remove the given row and column of a square matrix.

    Args:
        m (list): matrix.
        row (int): row to omit.
        col (int): column to omit.

    Returns:
        list: the matrix with the omitted row and column.
    """
    return [[m[i][j] for j in range(len(m[i])) if j != col]
            for i in range(len(m)) if i != row]


def determinant(matrix):
    """Calculates the determinant of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        float: the determinant.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix[0])):
        omited_matrix = minor_m(matrix, 0, j)
        det += matrix[0][j] * ((-1) ** j) * determinant(omited_matrix)

    return det


def transponse(m):
    """Transpose a matrix

    Args:
        m (list): matrix.

    Returns:
        list: the transposed matrix
    """
    return [[m[row][col] for row in range(len(m))]
            for col in range(len(m[0]))]


def adjugate(matrix):
    """Calculates the adjugate of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        list: the adjugate.
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    if n == 1 and len(matrix[0]) == 1:
        return [[1]]

    cofactor = [[((-1) ** (i + j)) * determinant(minor_m(matrix, i, j))
                 for j in range(len(matrix[i]))] for i in range(len(matrix))]

    return transponse(cofactor)
