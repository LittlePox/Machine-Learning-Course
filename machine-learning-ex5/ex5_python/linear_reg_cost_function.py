import numpy as np

def linear_reg_cost_function(X, y, theta, lbda):
    m = X.shape[0]
    _theta = np.copy(theta).flatten()
    _theta[0] = 0
    residual = (y - X.dot(theta)).flatten()
    return np.inner(residual, residual) / m / 2 + np.inner(_theta, _theta) * lbda / m / 2

