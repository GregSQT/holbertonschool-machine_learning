#!/usr/bin/env python3
"""
Function used to plot a histogram of student scores for a project
"""


import numpy as np
import matplotlib.pyplot as plt


"""
Function without input
"""


def frequency():

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.title('Project A')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    bins = np.arange(0, 110, 10)
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 110, 10))
    plt.ylim(0, 30)
    plt.show()
