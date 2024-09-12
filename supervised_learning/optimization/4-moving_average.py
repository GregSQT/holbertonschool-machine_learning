#!/usr/bin/env python3
"""
Exercice 4 : Moving Average
"""


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a data set.

    Args:
        data (list): The list of data to calculate the moving average of.
        beta (float): The weight used for the moving average.

    Returns:
        list: A list containing the moving averages of the data.

    """
    mov_avgs = []  # List to store the moving averages
    mov_avg = 0  # Variable to hold the current moving average
    for i in range(len(data)):
        # Update the moving average iteratively
        mov_avg = beta * mov_avg + (1 - beta) * data[i]
        # Apply bias correction and add the moving average to the list
        mov_avgs.append(mov_avg / (1 - beta ** (i + 1)))
    return mov_avgs
