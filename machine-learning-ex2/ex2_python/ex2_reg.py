import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from plot_data import plot_data
from cost_function_reg import cost_function_reg
from gradient_reg import gradient_reg
from plot_decision_boundary import plot_decision_boundary
from sigmoid import sigmoid

def map_feature(X):
    x1 = X[:,0]
    x2 = X[:,1]
    degree = 6
    pos = 1
    out = np.ones((len(x1), 1))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            out = np.insert(out, pos, x1**(i-j) * x2**j, 1)
            pos = pos + 1
    return out

# Initialization
os.system('clear')
np.set_printoptions(4, suppress=True)

data = pd.read_csv('ex2data2.txt', header=None)
m = len(data)
X = data[[0, 1]].to_numpy().reshape((m, 2))
y = data[2].to_numpy().reshape((m, 1))

plot_data(X, y)
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 1')
plt.legend(['y=1', 'y=0'])

input('Program paused. Press enter to continue.')

# =========== Part 1: Regularized Logistic Regression ============

m, n = X.shape
X = map_feature(X)

initial_theta = np.zeros((X.shape[1], 1))
lbda = 1

cost = cost_function_reg(initial_theta, X, y, lbda)
grad = gradient_reg(initial_theta, X, y, lbda)


print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693')

print('Gradient at initial theta (zeros) - first five values only:', grad[0:5].flatten())
print('Expected gradients (approx) - first five values only:')
print(' 0.0085 0.0188 0.0001 0.0503 0.0115')

input('Program paused. Press enter to continue.')

test_theta = np.ones((X.shape[1], 1))

cost = cost_function_reg(test_theta, X, y, 10)
grad = gradient_reg(test_theta, X, y, 10)


print('Cost at test theta (with lambda = 10): ', cost)
print('Expected cost (approx): 3.16')

print('Gradient at initial theta (zeros) - first five values only:', grad[0:5].flatten())
print('Expected gradients (approx) - first five values only:')
print(' 0.3460 0.1614 0.1948 0.2269 0.0922')

input('Program paused. Press enter to continue.')

#============= Part 2: Regularization and Accuracies =============

initial_theta = np.zeros((X.shape[1], 1))
lbda = 100000

fmin_result = fmin_bfgs(cost_function_reg, initial_theta, gradient_reg, (X, y, lbda), maxiter=40000, full_output=True)
theta = fmin_result[0].reshape((X.shape[1],1))
cost = fmin_result[1]

plot_decision_boundary(theta, X, y, map_feature, -1, 1.19, 56)

input('Program paused. Press enter to continue.')

plt.close('all')

p = sigmoid(X.dot(theta))
p = np.round(p)

print('Train Accuracy: {:.2%}'.format(np.mean(p == y)))
print('Expected accuracy (with lambda = 1): 83.1% (approx)')