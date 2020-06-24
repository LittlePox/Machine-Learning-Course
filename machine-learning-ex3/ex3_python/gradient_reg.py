import numpy as np

from sigmoid import sigmoid

def gradient_reg(theta, X, y, lbda):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    _theta = np.copy(theta)
    _theta[0][0] = 0
    y_bar = sigmoid(X.dot(theta))
    grad = (X.T.dot(y_bar - y) + lbda * _theta) / m

    return grad.flatten()