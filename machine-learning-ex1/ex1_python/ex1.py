import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warm_up_exercise import warm_up_exercise
from plot_data import plot_data
from compute_cost import compute_cost
from gradient_descent import gradient_descent
from mpl_toolkits.mplot3d import Axes3D

# Initialization
os.system('clear')

#==================== Part 1: Basic Function ====================
# Complete warm_up_exercise.py
print('Running warm up exercise ... ')
print('5x5 Identity Matrix: ')

print(warm_up_exercise())

input('Program paused. Press enter to continue.')

#======================= Part 2: Plotting =======================
print('Plotting data...')
data = pd.read_csv('ex1data1.txt', header=None)
m = len(data)
X = data[0].to_numpy().reshape((m,1))
y = data[1].to_numpy().reshape((m,1))

# Plot Data
plot_data(data)

input('Program paused. Press enter to continue.')

#=================== Part 3: Cost and Gradient descent ===================

X = np.insert(X, 0, values=1, axis=1)
theta = np.array([[0],[0]])
iterations = 1500
alpha = 0.01

print('Testing the cost function ...')
J = compute_cost(X, y, theta)

print('With theta = [0 ; 0]\nCost computed = ', J)
print('Expected cost value (approx) 32.07')

J = compute_cost(X, y, np.array([[-1],[2]]))
print('With theta = [-1 ; 2]\nCost computed = ', J)
print('Expected cost value (approx) 54.24')

print('Running Gradient Descent ...')
theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: ', theta)
print('Expected theta values (approx): (-3.6303, 1.1664)')

plt.plot(np.delete(X, 0, axis=1), X.dot(theta), '-')
plt.legend(['Linear regression', 'Training data'])
plt.show()

input('Program paused. Press enter to continue.')

plt.close()

predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of', predict1[0] * 10000)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of', predict2[0] * 10000)

input('Program paused. Press enter to continue.')

#============= Part 4: Visualizing J(theta_0, theta_1) =============

print('Visualizing J(theta_0, theta_1) ...')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

for i in range(0, len(theta0_vals)):
    for j in range(0, len(theta1_vals)):
        t = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[i][j] = compute_cost(X, y, t)

fig = plt.figure()
ax = plt.axes(projection = '3d')

T0, T1 = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(T0, T1, J_vals.T, cmap="rainbow")
ax.set_xlabel('θ0')
ax.set_ylabel('θ1')
ax.set_zlabel('J')

plt.show()
plt.close()
plt.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
plt.xlabel('θ0')
plt.ylabel('θ1')
plt.plot(theta[0][0], theta[1][0], c='r', marker='x', ms=10, lw=2);
plt.show()