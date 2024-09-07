#!/usr/bin/env python3
"""
Function used to slices a matrix along specific axes
"""


def np_slice(matrix, axes={}):
    """
    Input 1 : a matrix
    Input 2 : the axis of the slice
    Return : The result of the slice
    """
    slice_result = []
    for item_nb in range(len(matrix.shape)):
        flag = 0
        for ind, ax in axes.items():
            if ind == item_nb:
                slice_result.append(slice(*ax))
                flag = 1
                break
        if flag == 0:
            slice_result.append(slice(None, None, None))
    return matrix[tuple(slice_result)]
