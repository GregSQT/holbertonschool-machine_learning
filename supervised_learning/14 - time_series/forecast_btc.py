#!/usr/bin/env python3
"""Forecast BTC with an RNN model"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_rnn_model(input_shape):
    """
    Creates an RNN model for BTC forecasting.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.SimpleRNN(50, activation='relu', return_sequences=False),
        layers.Dense(1)
        ])

    model.compile(optimizer='adam', loss='mse')
    return model

def plot_training_history(history):
    """
    Plots and saves training and validation loss over epochs.
    """
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')  # Save the plot as a PNG file
    plt.show()

def main():
    # Load preprocessed data
    X = np.load('X.npy')
    y = np.load('y.npy')

    # Split data into training and validation sets
    split_index = int(0.8 * len(X))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Create tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

    # Create and train model
    input_shape = (X_train.shape[1], 1)
    model = create_rnn_model(input_shape)

    # Train the model and capture the training history
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # Plot training history
    plot_training_history(history)

    # Save the trained model
    model.save('btc_rnn_model.h5')

if __name__ == "__main__":
    main()
