#!/usr/bin/env python3
"""
Function used to transpose a matrix
"""


def matrix_transpose(matrix):
    """
    Input : a matrix
    Return : The matrix flipped
    """
    trans = []
    for row in range(len(matrix[0])):
        trans.append([matrix[col][row] for col in range(len(matrix))])
    return trans
