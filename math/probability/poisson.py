#!/usr/bin/env python3
"""Module with Poisson class"""


class Poisson:
    """Class that represents a Poisson distribution"""

    EULER_NUMBER = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.

        Args:
            data: List of data to estimate the distribution (default: None).
            lambtha: Expected number of occurrences in a given time frame.

        Raises:
                ValueError: If lambtha is not a positive value.
                TypeError: If data is not a list.
                ValueError: If data does not contain at least two data points.
        """
        if data is None:
            self.set_lambtha(lambtha)
        else:
            self.set_lambtha(self.calculate_lambtha(data))

    def set_lambtha(self, lambtha):
        """
        Set the lambtha attribute.

        Args:
            lambtha: Expected number of occurrences in a given time frame.

        Raises:
            ValueError: If lambtha is not a positive value.
        """
        if lambtha <= 0:
            raise ValueError("lambtha must be a positive value")
        self.lambtha = float(lambtha)

    def calculate_lambtha(self, data):
        """
        Calculate the lambtha value based on the given data.

        Args:
            data: List of data points.

        Returns:
                float: The calculated lambtha value.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data does not contain at least two data points.
        """
        if not isinstance(data, list):
            raise TypeError("data must be a list")
        if len(data) < 2:
            raise ValueError("data must contain multiple values")
        total = sum(data)
        count = len(data)
        return float(total) / count

    @staticmethod
    def factorial(n):
        """Calculates factorial of given number

        Args:
            n: input number

        Returns:
            response of n factorial
        """
        if n == 0:
            return 1
        return n * Poisson.factorial(n - 1)

    def pmf(self, k):
        """Calculates Probability Mass Function (PMF)

        Args:
            k: number of successes

        Returns:
            PMF of k or 0 if k is out of range.
        """
        k = int(k)

        if k < 0:
            return 0

        num = (self.lambtha ** k) * (self.EULER_NUMBER ** -self.lambtha)
        den = self.factorial(k)

        return num / den

    def cdf(self, k):
        """Calculates Cumulative Distribution Function (CDF)

        Args:
            k: number of successes

        Returns:
            CDF of k or 0 if k is out of range.
        """
        k = int(k)

        if k < 0:
            return 0

        summation = 0
        for i in range(k + 1):
            summation += (self.lambtha ** i) / (self.factorial(i))

        return (self.EULER_NUMBER ** -self.lambtha) * summation
