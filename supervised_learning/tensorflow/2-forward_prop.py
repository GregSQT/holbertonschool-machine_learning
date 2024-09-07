#!/usr/bin/env python3
"""
Exercice 2 - Forward Propagation
"""

import tensorflow.compat.v1 as tf
# Import the create_layer function from the 1-create_layer.py file
create_layer = __import__('1-create_layer').create_layer

def forward_prop(x, layer_sizes=[], activations=[]):
    """function that creates the forward propagation graph for the nn"""
    # Iterate over the layer_sizes and activations lists
    for i in range(len(layer_sizes)):
        # handle the first iteration of the loop
        if i == 0:
            # Initialize the prev variable with the value of x
            y_pred = x
        # Get the number of nodes in the current layer
        # Get the activation function for the current layer
        # Create the current layer using the create_layer function
        y_pred = create_layer(y_pred, layer_sizes[i], activations[i])
    return y_pred