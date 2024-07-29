#!/usr/bin/env python3
"""
Function used to calculate the shape of a matrix
"""


def matrix_shape(matrix):
    """
    Input : a matrix
    Return : The shape of the matrix
    """
    if type(matrix[0]) is not list:
        return [len(matrix)]
    else:
        return [len(matrix)] + matrix_shape(matrix[0])
