#!/usr/bin/env python3
"""
Multinormal
"""
import numpy as np
# Import the NumPy library for numerical operations.


class MultiNormal:
    """
    Class for Multivariate Normal distribution
    """

    def __init__(self, data):
        """
        Initializes a Multivariate Normal distribution.

        Args:
            data (np.ndarray): The dataset of shape (d, n).

        Raises:
            TypeError: If data is not a 2D numpy.ndarray.
            ValueError: If n is less than 2
            (data must contain multiple data points).

        Attributes:
            mean (np.ndarray): The mean vector of the dataset, shape (d, 1).
            cov (np.ndarray): The covariance matrix of
            the dataset, shape (d, d).
        """

        # Calculate mean and covariance when the instance is created
        self.mean, self.cov = self.calculate_mean_covariance(data)
        # Initialize the 'mean' and 'cov' attributes with
        # values calculated by the 'calculate_mean_covariance' method.

    def calculate_mean_covariance(self, data):
        """
        Calculates the mean and covariance matrix of a data set.

        Args:
            data (np.ndarray): The dataset of shape (d, n).

        Returns:
            np.ndarray: The mean vector of shape (d, 1).
            np.ndarray: The covariance matrix of shape (d, d).
        """

        # Check if data is a valid 2D numpy array
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        # Check if 'data' is a 2D numpy array; raise a TypeError if not.

        # Check if data contains multiple data points
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        # Check if 'data' contains multiple data points;
        # raise a ValueError if not.

        # Calculate the mean vector and centered data
        d, n = data.shape
        mean = np.mean(data, axis=1).reshape(-1, 1)
        centered_data = data - mean
        # Calculate the mean vector and centered data from 'data'.

        # Calculate the covariance matrix
        cov = np.matmul(centered_data, centered_data.T) / (n - 1)
        # Calculate the covariance matrix from the centered data.

        return mean, cov
        # Return the calculated mean and covariance.

    def pdf(self, x):
        """
        Calculates the Probability Density Function (PDF) at a data point.

        Args:
            x (np.ndarray): The data point of shape (d, 1).

        Raises:
            TypeError: If x is not a numpy.ndarray.
            ValueError: If x is not of shape (d, 1).

        Returns:
            float: The value of the PDF at the given data point.
        """

        # Check if x is a numpy array
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        # Check if 'x' is a numpy array; raise a TypeError if not.

        # Get the dimensionality of the covariance matrix
        d = self.cov.shape[0]
        # Get the dimensionality 'd' from the covariance matrix.

        # Check if x has the correct shape
        if x.ndim != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))
        # Check if 'x' has the correct shape; raise a ValueError if not.

        # Calculate determinant, inverse, and the exponent term
        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)
        x_minus_mean = x - self.mean
        exponent = -0.5 * np.matmul(x_minus_mean.T,
                                    np.matmul(cov_inv, x_minus_mean))
        # Calculate determinant, inverse, and the exponent term for the PDF.

        # Calculate the denominator and the PDF value
        denominator = 1.0 / np.sqrt(((2 * np.pi) ** d) * cov_det)
        pdf_value = denominator * np.exp(exponent)
        # Calculate the denominator and the PDF value.

        # Round the PDF value to match the expected value
        return round(float(pdf_value), 21)
        # Return the PDF value rounded to 16 decimal places.
