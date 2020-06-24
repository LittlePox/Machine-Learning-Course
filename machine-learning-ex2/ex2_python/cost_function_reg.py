import numpy as np

from sigmoid import sigmoid

def cost_function_reg(theta, X, y, lbda):
    m, n = X.shape
    eps = 1.4901161193847656e-08
    theta = theta.reshape((n, 1))
    _theta = np.copy(theta)
    _theta[0][0] = 0
    y_bar = sigmoid(X.dot(theta))
    J = ((y - 1).T.dot(np.log(1 - y_bar + eps)) - y.T.dot(np.log(y_bar + eps)) + 0.5 * lbda * _theta.T.dot(_theta)) / m
    J = J[0][0]
    
    return J