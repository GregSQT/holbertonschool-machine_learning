#!/usr/bin/env python3
"""
Function used to plot x and y as a line graph
"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Function without input
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.title('Exponential Decay of C-14')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.plot(x, y)
    plt.xlim(left=0, right=28650)
    plt.yscale('log')
    plt.show()
