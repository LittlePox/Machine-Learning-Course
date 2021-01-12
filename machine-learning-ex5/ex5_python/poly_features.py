import numpy as np

def poly_features(X, p):
    X_poly = np.zeros((len(X), p))
    X_poly[:, 0] = X[:, 0]
    for i in range(1, p):
        X_poly[:, i] = np.multiply(X_poly[:, i - 1], X[:, 0])
    return X_poly

