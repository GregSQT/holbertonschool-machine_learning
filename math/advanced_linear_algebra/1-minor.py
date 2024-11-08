#!/usr/bin/env python3
"""
Advanced Linear Algebra (Determinant and Minor)
"""


def minor_m(m, row, col):
    """ The the given row and column of a square matrix.

    Args:
        m (list): matrix.
        row (int): row to omit.
        col (int): column to omit.

    Returns:
        the matrix with the omitted row, column.
    """
    return [[m[i][j] for j in range(len(m[i])) if j != col]
            for i in range(len(m)) if i != row]


def minor(matrix):
    """
    Calculates the minor matrix of a square matrix.

    Args:
        matrix (list of lists): The input matrix.

    Returns:
        list of lists: The minor matrix.
    """
    # Check if matrix is a list of lists
    if not isinstance(
        matrix,
        list) or not all(
        isinstance(
            row,
            list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Get the size (number of rows) of the matrix
    n = len(matrix)

    # Check if the matrix is square and non-empty
    if n == 0 or len(matrix[0]) != n:
        raise ValueError("matrix must be a non-empty square matrix")

    # Handle the case of a 1x1 matrix
    if n == 1:
        return [[1]]

    # Initialize an empty list for the minor matrix
    minor_mat = []

    # Iterate through the rows of the matrix
    for i in range(n):
        # Initialize an empty list for the current row of the minor matrix
        minor_row = []
        # Iterate through the columns of the matrix
        for j in range(n):
            # Create a submatrix by excluding the current row and column
            submatrix = [row[:j] + row[j + 1:]
                         for row in (matrix[:i] + matrix[i + 1:])]
            # Calculate the determinant using the provided code
            minor_row.append(determinant(submatrix))
        # Append the minor row to the minor matrix
        minor_mat.append(minor_row)

    # Return the minor matrix
    return minor_mat


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.

    Args:
        matrix (list): matrix to calculate.

    Returns:
        float: the determinant.
    """
    # Check if the input is a list and not empty
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if each element of the input is a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Get the size (number of rows/columns) of the matrix
    n = len(matrix)

    # Check if the matrix is square and non-empty
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Base case: 1x1 matrix, return the single element
    if n == 1:
        return matrix[0][0]

    # Base case: 2x2 matrix, return the determinant formula
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Initialize the determinant value
    det = 0
    # Iterate through the columns of the matrix
    for j in range(len(matrix[0])):
        # Create a submatrix by excluding the current row and column
        omited_matrix = minor_m(matrix, 0, j)
        # Calculate the determinant using recursive calls
        det += matrix[0][j] * ((-1) ** j) * determinant(omited_matrix)

    return det
