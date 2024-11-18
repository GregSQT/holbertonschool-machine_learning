#!/usr/bin/env python3
"""
Advanced Linear Algebra determinant
"""

# Function to omit the given row and column of a square matrix


def minor_m(m, row, col):
    """
    Removes the given row and column of a square matrix.

    Args:
        m (list): matrix.
        row (int): row to omit.
        col (int): column to omit.

    Returns:
        The matrix with the omitted row and column.
    """
    # Create a new matrix without the specified row and column
    return [[m[i][j] for j in range(len(m[i])) if j != col]
            for i in range(len(m)) if i != row]

# Function to calculate the determinant of a square matrix


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

# Function to calculate the minor matrix of a matrix


def minor(matrix):
    """
    Calculates the minor matrix of a matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        The minor matrix.
    """
    # Check if the input is a valid list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if all elements are lists
    if all([isinstance(i, list) for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is square and non-empty
    if (len(matrix) == 0 or len(matrix) != len(matrix[0])) \
            or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    # Create a list for the minor matrix
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    # Calculate the minor matrix
    return [[((-1) ** (i + j)) * determinant(minor_m(matrix, i, j))
             for j in range(len(matrix[i]))] for i in range(len(matrix))]

# Function to calculate the cofactor matrix of a matrix


def cofactor(matrix):
    """
    Calculates cofactor matrix of a matrix.

    Args:
        (list): matrix to calculate.

    Returns:
        The cofactor matrix.
    """
    # Check if the input is a valid list of lists
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if all elements are lists
    if all([isinstance(i, list) for i in matrix]) is False:
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is square and non-empty
    if (len(matrix) == 0 or len(matrix) != len(matrix[0])) \
            or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    # Handle base case for 1x1 matrix
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    # Calculate the minor matrix
    minors = minor(matrix)
    height = len(minors)
    width = len(minors[0])

    # Initialize the cofactor matrix
    cofactor = [[0 for _ in range(width)] for _ in range(height)]

    # Calculate the cofactor matrix values
    for i in range(height):
        for j in range(width):
            sign = (-1) ** (i + j)
            cofactor[i][j] = sign * determinant(minor_m(matrix, i, j))

    return cofactor
