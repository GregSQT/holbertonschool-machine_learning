#!/usr/bin/env python3
import matplotlib.pyplot as plt  # Importing the matplotlib.pyplot module for plotting
import numpy as np  # Importing the numpy module for numerical computations
import tensorflow as tf  # Importing the tensorflow module for deep learning
lenet5 = __import__('4-lenet5').lenet5  # Importing the lenet5 function from the specified module

if __name__ == "__main__":
    # Execute the following code only if the script is run directly and not imported as a module

    np.random.seed(0)  # Setting the random seed for reproducibility
    tf.set_random_seed(0)  # Setting the random seed for reproducibility in TensorFlow

    # Load the MNIST dataset from the specified file
    lib = np.load('data/MNIST.npz')
    X_train = lib['X_train']  # Get the training images from the loaded dataset
    Y_train = lib['Y_train']  # Get the training labels from the loaded dataset
    X_valid = lib['X_valid']  # Get the validation images from the loaded dataset
    Y_valid = lib['Y_valid']  # Get the validation labels from the loaded dataset
    m, h, w = X_train.shape  # Get the number of examples, height, and width of the training images
    X_train_c = X_train.reshape((-1, h, w, 1))  # Reshape the training images to match the expected input shape
    X_valid_c = X_valid.reshape((-1, h, w, 1))  # Reshape the validation images to match the expected input shape

    x = tf.placeholder(tf.float32, (None, h, w, 1))  # Placeholder for the input images
    y = tf.placeholder(tf.int32, (None,))  # Placeholder for the target labels
    y_oh = tf.one_hot(y, 10)  # One-hot encoding of the target labels
    y_pred, train_op, loss, acc = lenet5(x, y_oh)  # Build the LeNet-5 model and define the training operations and metrics

    batch_size = 32  # Size of each mini-batch
    epochs = 10  # Number of training epochs

    init = tf.global_variables_initializer()  # Initialize all the variables
    with tf.Session() as sess:
        sess.run(init)  # Run the initialization operation

        for epoch in range(epochs):
            cost, accuracy = sess.run((loss, acc), feed_dict={x: X_train_c, y: Y_train})  # Compute the cost and accuracy on the training set
            cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x: X_valid_c, y: Y_valid})  # Compute the cost and accuracy on the validation set
            print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epoch, cost, accuracy, cost_valid, accuracy_valid))

            p = np.random.permutation(m)  # Shuffle the indices of the training examples
            X_shuffle = X_train_c[p]  # Shuffle the training images
            Y_shuffle = Y_train[p]  # Shuffle the training labels

            for i in range(0, m, batch_size):
                X_batch = X_shuffle[i:i+batch_size]  # Get the mini-batch of training images
                Y_batch = Y_shuffle[i:i+batch_size]  # Get the mini-batch of training labels
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})  # Perform a training step using the mini-batch

        cost, accuracy = sess.run((loss, acc), feed_dict={x: X_train_c, y: Y_train})  # Compute the final cost and accuracy on the training set
        cost_valid, accuracy_valid = sess.run((loss, acc), feed_dict={x: X_valid_c, y: Y_valid})  # Compute the final cost and accuracy on the validation set
        print("After {} epochs: {} cost, {} accuracy, {} validation cost, {} validation accuracy".format(epochs, cost, accuracy, cost_valid, accuracy_valid))

        Y_pred = sess.run(y_pred, feed_dict={x: X_valid_c, y: Y_valid})  # Get the predicted labels for the validation set
        print(Y_pred[0])
        Y_pred = np.argmax(Y_pred, 1)  # Convert the one-hot encoded predictions to class labels
        plt.imshow(X_valid[0])  # Display the first validation image
        plt.title(str(Y_valid[0]) + ' : ' + str(Y_pred[0]))  # Set the title of the plot with the true label and predicted label
        plt.show()  # Show the plot
        