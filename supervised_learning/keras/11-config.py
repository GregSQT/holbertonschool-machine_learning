#!/usr/bin/env python3
"""
Exercice 11 : Save and Load Configuration
"""
import tensorflow.keras as K


def save_config(network, filename):
    """Saves a model's configuration in JSON format"""
    config = network.to_json()  # Convert the model's configuration to JSON
    with open(filename, 'w') as f:  # Open the specified file in write mode
        f.write(config)  # Write the model's configuration to the file
    return None  # Return None as function doesn't need to return any value


def load_config(filename):
    """Loads a model with a specific configuration"""
    with open(filename, 'r') as f:  # Open the specified file in read mode
        config = f.read()  # Read the contents of the file
    # Create a model from the loaded JSON configuration
    network = K.models.model_from_json(config)
    return network  # Return the loaded network object
