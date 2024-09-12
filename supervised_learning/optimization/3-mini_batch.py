#!/usr/bin/env python3
"""
Exercice 3 : Mini-Batch
"""
import tensorflow as tf


shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train: numpy.ndarray of shape (m, 784) containing the training data
        Y_train: one-hot numpy.ndarray of shape (m, 10) containing training
        X_valid: numpy.ndarray of shape (m, 784) containing the validation data
        Y_valid: one-hot numpy.ndarray of shape (m, 10) containing validation
        batch_size: number of data points in a batch
        epochs: number of times the training should pass through whole dataset
        load_path: path from which to load the model
        save_path: path to where the model should be saved after training

    Returns:
        The path where the model was saved.
    """

    with tf.Session() as sess:
        # Restore the saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        # Get the necessary variables from the saved model
        var_names = ['x', 'y', 'accuracy', 'loss', 'train_op', 'y_pred']
        for var_name in var_names:
            globals()[var_name] = tf.get_collection(var_name)[0]

        # Iterate through the specified number of epochs
        for epoch in range(epochs + 1):

            # Evaluate the model on training and validation data
            loss_t = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            acc_t = sess.run(accuracy, feed_dict={x: X_train, y: Y_train})
            loss_v = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
            acc_v = sess.run(accuracy, feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(loss_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(loss_v))
            print("\tValidation Accuracy: {}".format(acc_v))

            if epoch < epochs:
                # Shuffle the full data set before each new epoch
                X_shuff, Y_shuff = shuffle_data(X_train, Y_train)

                # Define the iteration range for gradient descent
                batches_float = X_train.shape[0] / batch_size
                batches_int = int(X_train.shape[0] / batch_size)

                # Gradient descent step
                step = 0  # Reinitialize step to 0 between epochs

                for i in range(0, batches_int + 1):
                    step += 1

                    # Important: Make copies of X_shuff and Y_shuff
                    if i == batches_int:
                        if batches_float > batches_int:
                            X_batch = X_shuff[i * batch_size:]
                            Y_batch = Y_shuff[i * batch_size:]
                        else:
                            break
                    else:
                        X_batch = X_shuff[i * batch_size: (i + 1) * batch_size]
                        Y_batch = Y_shuff[i * batch_size: (i + 1) * batch_size]

                    # Pass the copies to feed_dict / train_op
                    sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    # Print after every 100 gradient descent steps
                    if step % 100 == 0:
                        loss_b = sess.run(
                            loss, feed_dict={
                                x: X_batch, y: Y_batch})
                        acc_b = sess.run(
                            accuracy, feed_dict={
                                x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(step))
                        print("\t\tCost: {}".format(loss_b))
                        print("\t\tAccuracy: {}".format(acc_b))

        # Save the trained model
        save_path = loader.save(sess, save_path)

    return save_path
