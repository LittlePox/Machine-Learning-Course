import numpy as np

from sigmoid import sigmoid

def cost_function(theta, X, y):
    m, n = X.shape
    eps = 1.4901161193847656e-08
    theta = theta.reshape((n, 1))
    y_bar = sigmoid(X.dot(theta))
    np.set_printoptions(4, suppress=True)
    J = ((y - 1).T.dot(np.log(1 - y_bar + eps)) - y.T.dot(np.log(y_bar + eps))) / m
    J = J[0][0]
    
    return J