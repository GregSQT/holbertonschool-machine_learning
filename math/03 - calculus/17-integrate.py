#!/usr/bin/env python3
"""
A function that calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Input 1 : poly : list of coefficients representing a polynomial
    Input 2 : C : an integer representing the integration constant
    Result : The integral of a polynomial
    """

    if type(poly) is not list or len(poly) == 0:
        return None
    elif type(C) is not int:
        return None
    else:
        if poly == [0]:
            return [C]
        exponent = 0
        integral = poly.copy()
        for i in range(len(integral)):
            if type(integral[i]) is int or type(integral[i]) is float:
                exponent += 1
                number = integral[i] / exponent
                integral[i] = int(number) if number % 1 == 0 else number
            else:
                return None
        integral.insert(0, C)
        return integral
