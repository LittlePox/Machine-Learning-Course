import os
import numpy as np
from scipy.io import loadmat

from cost_function_reg import cost_function_reg
from gradient_reg import gradient_reg
from one_vs_all import one_vs_all
from predict_one_vs_all import predict_one_vs_all

# Initialization
os.system('clear')
np.set_printoptions(4, suppress=True)

num_layer_size = 400
num_labels = 10

# =========== Part 1: Loading and Visualizing Data =============

annots = loadmat('ex3data1.mat')
X = annots['X']
y = annots['y']
m = X.shape[0]

# ============ Part 2a: Vectorize Logistic Regression ============

print('Testing lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2]).reshape(4, 1)
X_t = np.array(range(1,16)).reshape(3, 5).T / 10
X_t = np.insert(X_t, 0, 1, 1)
y_t = np.array([1,0,1,0,1]).reshape(5, 1)
lbda_t = 3

J = cost_function_reg(theta_t, X_t, y_t, lbda_t)
grad = gradient_reg(theta_t, X_t, y_t, lbda_t)

print('Cost: ', J)
print('Expected cost: 2.534819')
print('Gradients:')
print(grad)
print('Expected gradients:\n')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')

# ============ Part 2b: One-vs-All Training ============

lbda = 0.1
all_theta = one_vs_all(X, y, num_labels, lbda)

input('Program paused. Press enter to continue.')

# ================ Part 3: Predict for One-Vs-All ================

pred = predict_one_vs_all(all_theta, X)

print('Train Accuracy: {:.2%}'.format(np.mean(pred == y.flatten())))
