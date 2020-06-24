import numpy as np

from cost_function_reg import cost_function_reg
from gradient_reg import gradient_reg
from scipy.optimize import fmin_bfgs

def one_vs_all(X, y, num_labels, lbda):
    X = np.insert(X, 0, 1, 1)
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n))
    for i in range(0, num_labels):
        Y = (y == i + 1)
        initial_theta = np.zeros((n, 1))
        fmin_result = fmin_bfgs(cost_function_reg, initial_theta, gradient_reg, (X, Y, lbda), maxiter=50, full_output=True)
        theta = fmin_result[0].reshape((n,1))
        all_theta[i] = theta.flatten()
    return all_theta
