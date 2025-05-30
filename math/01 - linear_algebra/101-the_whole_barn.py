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


def elements_addition(mat1, mat2):
    """
    Input : 2 matrix
    Output : the recursive addition of the matrix element-wise
    """
    new_matrix = []
    if (type(mat1) and type(mat2)) == list:
        for i in range(len(mat1)):
            if type(mat1[i]) is not list:
                new_matrix.append(mat1[i] + mat2[i])
            else:
                new_matrix.append(elements_addition(mat1[i], mat2[i]))
        return new_matrix


def add_matrices(mat1, mat2):
    """
    Input : 2 matrix
    Output : the addition of the matrix element-wise
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    else:
        new_matrix = elements_addition(mat1, mat2)
        return new_matrix
