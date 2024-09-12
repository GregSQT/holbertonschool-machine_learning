#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer

if __name__ == '__main__':
    # Load MNIST data
    lib= np.load('data/MNIST.npz')
    X_3D = lib['X_train']
    X = X_3D.reshape((X_3D.shape[0], -1))

    # set random seed
    tf.set_random_seed(0)
    
    # Create a placeholder for input data
    x = tf.placeholder(tf.float32, shape=[None, 784])
    # Apply batch normalization using the create_batch_norm_layer function
    a = create_batch_norm_layer(x, 256, tf.nn.tanh)
    # Initialize variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        # Print the result of applying batch normalization on the first 5 samples of the input data
        print(sess.run(a, feed_dict={x:X[:5]}))
        