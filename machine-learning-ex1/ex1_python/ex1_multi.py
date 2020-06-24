import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from feature_normalize import feature_normalize
from gradient_descent import gradient_descent

# Initialization
os.system('clear')

np.set_printoptions(2, suppress=True)

print('Loading data...')
data = pd.read_csv('ex1data2.txt', header=None)
m = len(data)
X = data[[0,1]].to_numpy().reshape((m, 2))
y = data[[2]].to_numpy().reshape((m, 1))

print('First 10 examples from the dataset: ')
print('X =')
print(X[0:10])
print('y =')
print(y[0:10])

input('Program paused. Press enter to continue.')

#================ Part 1: Feature Normalization ================

print('Normalizing Features ...')

X, mu, sigma = feature_normalize(X)
X = np.insert(X, 0, values=1, axis=1)

#================ Part 2: Gradient Descent ================

print('Running gradient descent ...')
alpha = 0.01
num_iters = 400

theta = np.zeros((3, 1))
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
plt.plot(range(0,num_iters), J_history, '-', c='b', lw=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

print('Theta computed from gradient descent:')
print(theta)

# Estimate the price of a 1650 sq-ft, 3 br house

x = np.array([[1650, 3]])
x = (x - mu)/sigma
x = np.insert(x, 0, values=1, axis=1)

price = x.dot(theta)[0][0]

print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', price)

input('Program paused. Press enter to continue.')

#================ Part 3: Normal Equations ================

X = data[[0,1]].to_numpy().reshape((m, 2))
X = np.insert(X, 0, values=1, axis=1)

theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)

print('Theta computed from normal equations:')
print(theta)

price = np.array([1, 1650, 3]).dot(theta)[0]

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', price)