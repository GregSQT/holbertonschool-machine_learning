#!/usr/bin/env python3
"""Module with Binomial class"""


class Binomial:
    """Class that represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor"""
        # If data is provided, calculate n and p based on the data
        if data is not None:
            n, p = self.calculate_n_p(data)
        # Set the instance attributes for n and p
        self.n = n
        self.p = p

    @property
    def n(self):
        """Getter for n"""
        return self.__n

    @n.setter
    def n(self, n):
        """Setter for n"""
        # Check if n is a positive value
        if n <= 0:
            raise ValueError('n must be a positive value')
        self.__n = int(n)

    @property
    def p(self):
        """Getter for p"""
        return self.__p

    @p.setter
    def p(self, p):
        """Setter for p"""
        # Check if p is between 0 and 1 exclusive
        if not 0 < p < 1:
            raise ValueError('p must be greater than 0 and less than 1')
        self.__p = float(p)

    @staticmethod
    def factorial(n):
        """Calculates the factorial of a given number"""
        factorial_n = 1
        for i in range(1, n + 1):
            factorial_n *= i
        return factorial_n

    @classmethod
    def calculate_n_p(cls, data):
        """Calculates the values of n and p based on the given data"""
        if not isinstance(data, list):
            raise TypeError('data must be a list')
        if len(data) < 2:
            raise ValueError('data must contain multiple values')

        len_data = len(data)
        mean = sum(data) / len_data
        variance = sum([(number - mean) ** 2 for number in data]) / len_data
        # Calculate p using the formula 1 - (variance / mean)
        p = 1 - (variance / mean)
        # Calculate n using the formula round(mean / p)
        n = int(round(mean / p))
        # Update p using the formula mean / n
        p = (mean / n)
        return n, p

    def pmf(self, k):
        """Calculates Probability Mass Function (PMF)
        Args:
            k: number of successes
        Returns:
            PMF of k or 0 if k is out of range.
        """
        # Convert k to an integer
        k = int(k)
        # Check if k is out of range
        if k < 0 or k > self.n:
            return 0
        # Calculate the binomial coefficient using the get_bcf method
        binomial_coefficient = self.get_bcf(k)
        # Calculate the complement probability q
        q = 1 - self.p
        # Calculate the PMF using the binomial
        return binomial_coefficient * ((self.p ** k) * (q ** (self.n - k)))

    def cdf(self, k):
        """Calculates Cumulative Distribution Function (CDF)

        Args:
            k: number of successes

        Returns:
            CDF of k or 0 if k is out of range.
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum([self.cdf_equation(i) for i in range(k + 1)])

    def get_bcf(self, k):
        """Calculates binomial coefficient with a given number"""
        # Calculate the factorial of n
        n_factorial = self.factorial(self.n)
        # Calculate the factorial of k
        k_factorial = self.factorial(k)
        # Calculate the factorial of (n - k)
        n_k_factorial = self.factorial(self.n - k)
        # Calculate the binomial coefficient using the factorials
        binomial_coefficient = n_factorial / (n_k_factorial * k_factorial)
        return binomial_coefficient

    def cdf_equation(self, i):
        """Calculates cdf for each iteration"""
        # Calculate the value of r using the binomial
        r = self.get_bcf(i) * ((self.p ** i) * ((1 - self.p) ** (self.n - i)))
        return r
