#!/usr/bin/env python3
"""
Exercice 7 : This module generates a scatter plot of sampled
elevations on a mountain.
The x and y coordinates are randomly generated
and the z coordinate (elevation)
is calculated based on the x and y coordinates. T
he scatter plot uses a colorbar
to represent the elevation levels.
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    This function generates a scatter plot of sampled
    elevations on a mountain.
    The x and y coordinates are randomly generated and
    the z coordinate (elevation)
    is calculated based on the x and y coordinates. The
    scatter plot uses a colorbar
    to represent the elevation levels.
    """

    # Set the seed for the random number generator
    # for reproducibility
    np.random.seed(5)

    # Generate 2000 random x and y coordinates,
    # scaled by a factor of 10
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10

    # Calculate the z coordinate (elevation) based
    # on the x and y coordinates
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    # Create a new figure with a specific size
    plt.figure(figsize=(6.4, 4.8))

    # Create a scatter plot with the x and y coordinates,
    # using the z coordinate for color
    scatter = plt.scatter(x, y, c=z)

    # Add a colorbar to the plot, with a label
    plt.colorbar(scatter).set_label('elevation (m)')

    # Set the labels for the x and y axes, and the title of the plot
    plt.xlabel('x coordinate (m)')
    plt.ylabel('y coordinate (m)')
    plt.title('Mountain Elevation')

    # Display the plot
    plt.show()
    