#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""
Function used to draw a line
"""


def line():
    """
    No argument
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.xlim(0, 10)
    plt.plot(y, color='r')
    plt.show()
