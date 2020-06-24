import numpy as np

from sigmoid import sigmoid

def gradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y_bar = sigmoid(X.dot(theta))
    grad = (y_bar - y).T.dot(X) / m

    return grad.flatten()