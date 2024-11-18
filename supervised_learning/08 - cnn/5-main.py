#!/usr/bin/env python3
"""
Main file
"""
# Description of the purpose of the code

# Force Seed - fix for Keras
SEED = 0

import matplotlib.pyplot as plt  # Importing the matplotlib.pyplot module for plotting
import os  # Importing the os module for operating system functionalities
os.environ['PYTHONHASHSEED'] = str(SEED)  # Setting the PYTHONHASHSEED environment variable
import random  # Importing the random module for random number generation
random.seed(SEED)  # Setting the seed for random number generation
import numpy as np  # Importing the numpy module for numerical operations
np.random.seed(SEED)  # Setting the seed for numpy random number generation
import tensorflow as tf  # Importing the tensorflow module for deep learning
tf.set_random_seed(SEED)  # Setting the seed for tensorflow random number generation
import tensorflow.keras as K  # Importing the tensorflow.keras module for deep learning

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# Creating a tensorflow session configuration with single thread
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# Creating a tensorflow session with the specified session configuration
K.backend.set_session(sess)  # Setting the session for keras backend

lenet5 = __import__('5-lenet5').lenet5  # Importing the lenet5 function from module '5-lenet5'

if __name__ == "__main__":
    # Main program starts here

    lib = np.load('data/MNIST.npz')  # Loading the MNIST dataset
    X_train = lib['X_train']  # Loading the training data
    m, h, w = X_train.shape  # Getting the shape of the training data
    X_train_c = X_train.reshape((-1, h, w, 1))  # Reshaping the training data to 4D tensor (image width, image height, channels)
    Y_train = lib['Y_train']  # Loading the training labels
    Y_train_oh = K.utils.to_categorical(Y_train, num_classes=10)  # One-hot encoding the training labels
    X_valid = lib['X_valid']  # Loading the validation data
    X_valid_c = X_valid.reshape((-1, h, w, 1))  # Reshaping the validation data to 4D tensor (image width, image height, channels)
    Y_valid = lib['Y_valid']  # Loading the validation labels
    Y_valid_oh = K.utils.to_categorical(Y_valid, num_classes=10)  # One-hot encoding the validation labels
    X = K.Input(shape=(h, w, 1))  # Creating a keras input layer with the shape of the images
    model = lenet5(X)  # Building the LeNet-5 model using the lenet5 function
    batch_size = 32  # Setting the batch size for training
    epochs = 5  # Setting the number of epochs for training
    model.fit(X_train_c, Y_train_oh, batch_size=batch_size, epochs=epochs,
                       validation_data=(X_valid_c, Y_valid_oh))  # Training the model on the training data
    Y_pred = model.predict(X_valid_c)  # Predicting the labels for the validation data
    print(Y_pred[0])  # Printing the predicted label for the first validation sample
    Y_pred = np.argmax(Y_pred, 1)  # Getting the index of the predicted label for each validation sample
    plt.imshow(X_valid[0])  # Displaying the first validation image
    plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))  # Setting the title of the plot with true and predicted labels
    plt.show()  # Showing the plot
    