#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import zipfile
update_variables_Adam = __import__('9-Adam').update_variables_Adam

def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Update a variable in place using the Adam optimization algorithm.

    Args:
        alpha: Learning rate.
        beta1: Weight used for the first moment.
        beta2: Weight used for the second moment.
        epsilon: Small number to avoid division by zero.
        var: Numpy array containing the variable to be updated.
        grad: Numpy array containing the gradient of var.
        v: Previous first moment of var.
        s: Previous second moment of var.
        t: Time step used for bias correction.

    Returns:
        Updated variable, new first moment, and new second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    v_corrected = v / (1 - beta1 ** t)

    s = beta2 * s + (1 - beta2) * (grad ** 2)
    s_corrected = s / (1 - beta2 ** t)

    var -= alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v, s


def forward_prop(X, W, b):
    Z = np.matmul(X, W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


def calculate_grads(Y, A, W, b):
    m = Y.shape[0]
    dZ = A - Y
    dW = np.matmul(X.T, dZ) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    return dW, db


def calculate_cost(Y, A):
    m = Y.shape[0]
    loss = - (Y * np.log(A) + (1 - Y) * np.log(1 - A))
    cost = np.sum(loss) / m
    return cost


if __name__ == '__main__':
    zip_path = 'data/Binary_Train.zip'

    zip_file = zipfile.ZipFile(zip_path, 'r')

    npz_file = zip_file.open('Binary_Train.npz')

    lib_train = np.load(npz_file)
    X_3D, Y = lib_train['X'], lib_train['Y'].T
    X = X_3D.reshape((X_3D.shape[0], -1))

    nx = X.shape[1]
    np.random.seed(0)
    W = np.random.randn(nx, 1)
    b = 0
    dW_prev1 = np.zeros((nx, 1))
    db_prev1 = 0
    dW_prev2 = np.zeros((nx, 1))
    db_prev2 = 0
    for i in range(1000):
        A = forward_prop(X, W, b)
        if not (i % 100):
            cost = calculate_cost(Y, A)
            print('Cost after {} iterations: {}'.format(i, cost))
        dW, db = calculate_grads(Y, A, W, b)
        W, dW_prev1, dW_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, W, dW, dW_prev1, dW_prev2, i + 1)
        b, db_prev1, db_prev2 = update_variables_Adam(0.001, 0.9, 0.99, 1e-8, b, db, db_prev1, db_prev2, i + 1)
    A = forward_prop(X, W, b)
    cost = calculate_cost(Y, A)
    print('Cost after {} iterations: {}'.format(1000, cost))

    Y_pred = np.where(A >= 0.5, 1, 0)
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_3D[i])
        plt.title(str(Y_pred[i, 0]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()