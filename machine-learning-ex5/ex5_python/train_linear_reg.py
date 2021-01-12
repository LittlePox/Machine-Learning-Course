import numpy as np
from scipy.optimize import fmin_cg
from linear_reg_cost_function import cost_function
from linear_reg_cost_gradient import cost_gradient

def cost(theta, X, y, lbda):
    theta = np.reshape(theta, (X.shape[1], 1))
    return cost_function(X, y, theta, lbda)

def gradient(theta, X, y, lbda):
    theta = np.reshape(theta, (X.shape[1], 1))
    return cost_gradient(X, y, theta, lbda)

def train_linear_reg(X, y, lbda):
    init_theta = np.zeros((X.shape[1], 1))
    fmin_result = fmin_cg(cost, init_theta, gradient, (X, y, lbda), maxiter=50, full_output=True)
    return np.reshape(fmin_result[0], (X.shape[1], 1))


