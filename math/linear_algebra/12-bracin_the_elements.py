#!/usr/bin/env python3
"""
Function used to element-wise addition, subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """
    Input : 2 matrix (mat1 and mat2)
    Returns the result of the operations of the matrix
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
