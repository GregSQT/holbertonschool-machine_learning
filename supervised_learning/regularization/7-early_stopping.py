#!/usr/bin/env python3
"""
Exercice 7 : Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be
    stopped early based on validation cost.

    Arguments:
    - cost: Current validation cost of the neural network.
    - opt_cost: Lowest recorded validation cost of the neural network.
    - threshold: Threshold used for early stopping.
    - patience: Patience count used for early stopping.
    - count: Count of how long the threshold has not been met.

    Returns:
    - Tuple containing a boolean indicating
    whether to stop early and updated count value.
    """

    # Initialize early stopping flag as False
    early_stopping = False

    # Calculate the difference between optimal cost and current cost
    cost_difference = opt_cost - cost

    # Check if the cost difference is less than or equal to the threshold
    if cost_difference <= threshold:
        # Increment the count since the threshold has not been met
        count += 1
    else:
        # Reset the count since the threshold has been met
        count = 0

    # Check if the count has reached the patience limit
    if count == patience:
        # Early stopping condition met, set early stopping flag to True
        early_stopping = True
        return (early_stopping, count)

    # Return the early stopping flag and the updated count value
    return (early_stopping, count)
