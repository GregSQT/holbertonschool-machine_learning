#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Function used to draw a line
"""


def line():
    """
    No argument
    Will draw a line from 0 to 10 on axis
    the line will be a representation of cube of x
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.xlim(0, 10)
    plt.plot(y, color='r')
    plt.show()
