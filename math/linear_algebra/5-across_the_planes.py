#!/usr/bin/env python3
"""
Function used to sum 2 arrays element wise
"""


def add_arrays(arr1, arr2):
    """
    Input : 2 arrays
    Return : The sum of the arrays
    """
    if len(arr1) != len(arr2):
        return None
    return [(arr1[i] + arr2[i]) for i in range(len(arr1))]
