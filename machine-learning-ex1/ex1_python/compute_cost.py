import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    residual = y - X.dot(theta)
    residual = residual.T
    return np.inner(residual, residual)[0][0] / m / 2