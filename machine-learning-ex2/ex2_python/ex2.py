import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

from plot_data import plot_data
from cost_function import cost_function
from gradient import gradient
from plot_decision_boundary import plot_decision_boundary
from sigmoid import sigmoid

def map_feature(X):
    X = np.insert(X, 0, 1, 1)
    X = np.insert(X, n + 1, X[:,1]*X[:,1] / 100, 1)
    X = np.insert(X, n + 2, X[:,1]*X[:,2] / 100, 1)
    X = np.insert(X, n + 3, X[:,2]*X[:,2] / 100, 1)
    return X


# Initialization
os.system('clear')

data = pd.read_csv('ex2data1.txt', header=None)
m = len(data)
X = data[[0, 1]].to_numpy().reshape((m, 2))
y = data[2].to_numpy().reshape((m, 1))

#==================== Part 1: Plotting ====================

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plot_data(X, y)
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend(['Admitted', 'Not admitted'])

input('Program paused. Press enter to continue.')

#============ Part 2: Compute Cost and Gradient ============

m, n = X.shape
X = map_feature(X)

initial_theta = np.zeros((n + 4, 1))

cost = cost_function(initial_theta, X, y)
grad = gradient(initial_theta, X, y)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros)', grad)

test_theta = np.array([-24, 0.2, 0.2, 0, 0.1, 0]).reshape((6, 1))
cost = cost_function(test_theta, X, y)
grad = gradient(test_theta, X, y)

print('Cost at test theta:', cost)
print('Gradient at test theta', grad)

# ============= Part 3: Optimizing using fminunc  =============

fmin_result = fmin_bfgs(cost_function, initial_theta, gradient, (X, y), maxiter=400, full_output=True)
theta = fmin_result[0].reshape((6,1))
cost = fmin_result[1]

print('Cost at theta found by fminunc: ', cost, theta)

plot_decision_boundary(theta, X, y, map_feature)

plt.close('all')

# ============== Part 4: Predict and Accuracies ==============

prob = sigmoid(map_feature(np.array([[45, 85]])).dot(theta))[0][0]

print('For a student with scores 45 and 85, we predict an admission probability of', prob)

p = sigmoid(X.dot(theta))
p = np.round(p)

print('Train Accuracy: {:.2%}'.format(np.mean(p == y)))