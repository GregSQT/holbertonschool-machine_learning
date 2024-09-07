#!/usr/bin/env python3
"""
Exercice 7 : Train a model, with learning rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """function that trains a model using mini-batch gradient descent"""

    callbacks = []

    # Add early stopping callback
    # if validation data and early stopping are enabled
    if validation_data and early_stopping:
        # Create an EarlyStopping callback object
        # Monitor the validation loss and stop training if it doesn't improve
        # for a certain number of epochs (defined by the patience parameter)
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)

    # Add learning rate decay callback
    # if validation data and learning rate decay are enabled
    if validation_data and learning_rate_decay:

        # Define a schedule function that takes the epoch index as input
        # and returns a new learning rate as output
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        # Create a LearningRateScheduler
        # callback object with the defined schedule function
        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=schedule, verbose=1)
        callbacks.append(lr_decay)

    # Train the network using the fit method of the network object
    # Pass the data, labels, and other training parameters
    # Include the validation data and callbacks if provided
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)

    # Return the history object, which
    # contains training metrics and loss values
    return history
