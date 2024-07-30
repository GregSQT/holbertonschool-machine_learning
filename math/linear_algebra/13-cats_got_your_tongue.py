#!/usr/bin/env python3
"""
Function used to concatenates two matrices along a specific axis
"""


def np_cat(mat1, mat2, axis=0):
    """
    Input : a matrix
    Return : Concatenated matrixes along an axis
    """
    return np.concatenate((mat1, mat2), axis=axis)
