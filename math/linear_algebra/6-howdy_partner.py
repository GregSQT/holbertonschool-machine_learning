#!/usr/bin/env python3
"""
Function used to concatenates two arrays
"""


def cat_arrays(arr1, arr2):
    """
    Input :  2 arrays
    Return : The concatenation of the arrays
    """
    concat = arr1.copy()
    for i in arr2:
        concat.append(i)
    return concat
