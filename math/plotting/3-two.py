#!/usr/bin/env python3
"""
Function used to plot x and y as a line graph
"""


import numpy as np
import matplotlib.pyplot as plt


"""
Function without input
"""
def two():

    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    plt.title('Exponential Decay of Radioactive Elements')
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.xlim(left=0, right=20000)
    plt.ylim(bottom=0, top=1)
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')
    plt.legend()
    plt.show()
