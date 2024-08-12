#!/usr/bin/env python3
"""
A function that calculates a cuadratic sum
"""


def summation_i_squared(n):
    """
    Input : n (limit)
    Output : sum of i^2 with i from 1 to n
    """
    if n < 1 or type(n) is not int:
        return None
    else:
        iterations = range(1, n + 1)
    return sum(map(lambda n: pow(n, 2), iterations))
