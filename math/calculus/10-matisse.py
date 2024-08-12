#!/usr/bin/env python3
""" 
Function that calculates the derivative of a polynomial
"""


def poly_derivative(poly):
    """
    Find  the coefficients representing a polynomial
    """
    if type(poly) is not list or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    for x in range(1, len(poly)):
        poly[x] = poly[x] * x
    poly.pop(0)
    return poly
