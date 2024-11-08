#!/usr/bin/env python3
"""
Advanced Linear Algebra (Inverse Determinant)
"""
# Define a function to omit the given row and column of a square matrix.


def minor_m(m, row, col):
    """
    Remove given row and column of a square matrix.

    Args:
        m (list): The input square matrix.
        row (int): The row index to omit.
        col (int): The column index to omit.

    Returns:
        list: The matrix with the specified row and column removed.
    """
    return [[m[i][j] for j in range(len(m[i])) if j != col]
            for i in range(len(m)) if i != row]

# Define a function to calculate the determinant of a square matrix.


def determinant(matrix):
    """
    Calculate the determinant of a square matrix using recursion.

    Args:
        matrix (list): The input square matrix.

    Returns:
        float: The determinant of the matrix.
    """

    # Check if matrix is a list of lists and is not empty
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if each element of the matrix is a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Get the size of the matrix (number of rows/columns)
    n = len(matrix)

    # Check if it's a non-empty square matrix
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a non-empty square matrix")

    # Base case: If the matrix is 1x1, return its only element as the
    # determinant
    if n == 1:
        return matrix[0][0]

    # Base case: If the matrix is 2x2, calculate the determinant directly
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Initialize the determinant
    det = 0

    # Calculate the determinant using expansion by minors
    for j in range(len(matrix[0])):
        # Calculate the minor by omitting the first row and the j-th column
        omited_matrix = minor_m(matrix, 0, j)

        # Add the product of the element and its corresponding cofactor
        det += matrix[0][j] * ((-1) ** j) * determinant(omited_matrix)

    return det

# Define a function to transpose a matrix.


def transpose(m):
    """
    Transpose a given matrix.

    Args:
        m (list): The input matrix.

    Returns:
        list: The transposed matrix.
    """
    return [[m[row][col] for row in range(len(m))]
            for col in range(len(m[0]))]

# Define a function to calculate the adjugate of a square matrix.


def adjugate(matrix):
    """
    Calculate the adjugate of a square matrix.

    Args:
        matrix (list): The input square matrix.

    Returns:
        list: The adjugate matrix.
    """

    # Check if matrix is a list of lists and is not empty
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if each element of the matrix is a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Check if it's a non-empty square matrix, or [[]]
    if (matrix[0] and len(matrix) != len(matrix[0])) or \
            matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    # Base case: If the matrix is 1x1, return [[1]] as the adjugate
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    # Calculate the cofactors of the matrix
    cofactor = [[((-1) ** (i + j)) * determinant(minor_m(matrix, i, j))
                 for j in range(len(matrix[i]))] for i in range(len(matrix))]

    # Transpose the cofactor matrix to obtain the adjugate
    return transpose(cofactor)

# Define a function to calculate the inverse of a matrix.


def inverse(matrix):
    """
    Calculate the inverse of a square matrix if it exists.

    Args:
        matrix (list): The input square matrix.

    Returns:
        list: The inverse matrix or None if the matrix is singular.
    """

    # Check if matrix is a list of lists and is not empty
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    # Check if each element of the matrix is a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")

    # Check if it's a non-empty square matrix, or [[]]
    if (len(matrix) == 0 or len(matrix) != len(matrix[0])) \
            or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    # Check if any row has a different length
    if any([len(row) != len(matrix) for row in matrix]):
        raise ValueError("matrix must be a non-empty square matrix")

    # Calculate the determinant of the matrix
    det = determinant(matrix)

    # If the determinant is 0, the matrix is singular, return None
    if det == 0:
        return None

    # Calculate the adjugate matrix
    adjugated = adjugate(matrix)

    # Calculate the inverse by dividing each element of the adjugate matrix by
    # the determinant
    return [[n / det for n in row] for row in adjugated]
