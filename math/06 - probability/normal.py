#!/usr/bin/env python3
"""Module with Normal class"""


class Normal:
    """Class that represents a normal distribution"""

    EULER_NUMBER = 2.7182818285
    PI = 3.1415926536

    @staticmethod
    def get_stddev(data, mean):
        """Calculates Standard Deviation with a given data and mean"""
        summation = 0
        for number in data:
            summation += (number - mean) ** 2
        return (summation / len(data)) ** (1 / 2)

    @classmethod
    def get_erf(cls, x):
        """Calculates the Error Function (erf) for a given value x"""
        seq = (
            x - ((x ** 3) / 3)
            + ((x ** 5) / 10)
            - ((x ** 7) / 42)
            + ((x ** 9) / 216)
        )
        erf = (2 / (cls.PI ** (1 / 2))) * seq
        return erf

    def __init__(self, data=None, mean=0, stddev=1.):
        """Class constructor"""

        # Handling case when data is provided
        if data is not None:
            # Checking if data is a list
            if not isinstance(data, list):
                raise TypeError('data must be a list')

            # Checking if data contains at least two values
            if len(data) < 2:
                raise ValueError('data must contain multiple values')

            # Calculating the mean and standard deviation based on data
            self.mean = sum(data) / len(data)
            self.stddev = self.get_stddev(data, self.mean)
        else:
            # Handling case when data is not provided
            # Checking if stddev is a positive value
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')

            # Assigning the mean and standard deviation values as floats
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """Calculates the z-score for a given x-value."""
        # Calculate the z-score using the formula (x - mean) / stddev
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value for a given z-score."""
        # Calculate the x-value using the formula (z * stddev) + mean
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Calculates Probability Density Function (PDF)

        Args:
            x: x-value

        Returns:
            PDF
        """
        exp_component = (-1/2) * (((x - self.mean)/self.stddev) ** 2)
        euler_exp = self.EULER_NUMBER ** exp_component
        cdf = euler_exp/(self.stddev * ((2 * self.PI) ** (1/2)))
        return cdf

    def cdf(self, x):
        """Calculates the Cumulative Distribution Function (CDF) x-value."""
        # Calculate the standardized value
        # using (x - mean) / (stddev * sqrt(2))
        expression = (x - self.mean) / (self.stddev * (2 ** (1/2)))
        # Use the error function to calculate the CDF and return the result
        return (1/2) * (1 + self.get_erf(expression))
