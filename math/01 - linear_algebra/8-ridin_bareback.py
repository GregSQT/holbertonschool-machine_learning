#!/usr/bin/env python3
"""
Function used to performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Input : 2 matrix
    Return : Multiplication of the 2 matrix as a new matrix
    """
    if len(mat1[0]) == len(mat2):
        result_matrix = []
        for row in range(len(mat1)):
            new_row = []
            for col2 in range(len(mat2[0])):
                number = 0
                for col1 in range(len(mat1[0])):
                    number += (mat1[row][col1] * mat2[col1][col2])
                new_row.append(number)
            result_matrix.append(new_row)
        return result_matrix
    else:
        return None
