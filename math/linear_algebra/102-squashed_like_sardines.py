#!/usr/bin/env python3
"""
Function used to adds two matrices
"""


def matrix_shape(matrix):
    """
    Input : a matrix
    Returns the shape of the matrix
    """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])


def elements_concatenate(mat1, mat2, axis):
    """
    Needs a matrix as input
    Returns a concatenated matrix
    """
    new_matrix = []
    if axis == 0:
        new_matrix = mat1 + mat2
        return new_matrix
    for i in range(len(mat1)):
        new_matrix.append(elements_concatenate(mat1[i], mat2[i], axis - 1))
    return new_matrix


def cat_matrices(mat1, mat2, axis=0):
    """
    Input : 2 matrix
    Output : a concatenation of the 2 matrix along a specific axis
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        new_matrix = elements_concatenate(mat1, mat2, axis)
        return new_matrix
