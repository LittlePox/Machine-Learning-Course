import numpy as np

def cost_gradient(X, y, theta, lbda):
    m = X.shape[0]
    _theta = np.copy(theta).flatten()
    _theta[0] = 0
    residual = (y - X.dot(theta)).flatten()
    return _theta.T * lbda / m - residual.T.dot(X) / m

