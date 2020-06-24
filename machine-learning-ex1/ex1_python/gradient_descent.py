import numpy as np
from compute_cost import compute_cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)

    for i in range(0, iterations):
        theta = theta - alpha / m * (X.T.dot(X.dot(theta) - y))
        J_history[i] = compute_cost(X, y, theta)
    
    return theta, J_history
