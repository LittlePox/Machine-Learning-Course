import numpy as np

def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 0.0001
    for p in range(theta.size):
        perturb[p] = e
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (e + e)
        perturb[p] = 0
    return numgrad