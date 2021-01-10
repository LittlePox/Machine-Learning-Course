import os
import numpy as np
from scipy.io import loadmat
from nn_cost import nn_cost
from nn_grad import nn_grad
from sigmoid_gradient import sigmoid_gradient
from rand_initialize_weights import rand_initialize_weights
from check_nn_gradients import check_nn_gradients
from scipy.optimize import fmin_cg
from predict import predict

# Initialization
os.system('clear')
np.set_printoptions(4, suppress=True)

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# =========== Part 1: Loading and Visualizing Data =============

annots = loadmat('ex4data1.mat')
X = annots['X']
y = annots['y']
m = X.shape[0]

# ================ Part 2: Loading Pameters ================

input('Loading Saved Neural Network Parameters ...')

annots = loadmat('ex4weights.mat')
Theta1 = annots['Theta1']
Theta2 = annots['Theta2']

nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))

# ================ Part 3: Compute Cost (Feedforward) ================

print('Feedforward Using Neural Network ...')

lbda = 0
J = nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbda)
print('Cost at parameters (loaded from ex4weights):', J)
print('(this value should be about 0.287629)')

# =============== Part 4: Implement Regularization ===============

lbda = 1
J = nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbda)
print('Cost at parameters with regularization (loaded from ex4weights):', J)
print('(this value should be about 0.383770)')

input('Program paused. Press enter to continue.')

# ================ Part 5: Sigmoid Gradient  ================

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:', g)

input('Program paused. Press enter to continue.')

# ================ Part 6: Initializing Pameters ================

print('Initializing Neural Network Parameters ...')

initial_Theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
initial_Theta2 = rand_initialize_weights(hidden_layer_size, num_labels)

initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))

# =============== Part 7: Implement Backpropagation ===============

print('Checking Backpropagation...')
check_nn_gradients()

# =============== Part 8: Implement Regularization ===============

print('Checking Backpropagation (w/ Regularization)...')
lbda = 3
check_nn_gradients(lbda)

debug_j = nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbda)

print('Cost at (fixed) debugging parameters (w/ lambda = 3): ', debug_j)
print('(for lambda = 3, this value should be about 0.576051)')

input('Program paused. Press enter to continue.')

# =============== Part 9: Training NN ===============

print("Training Neural Network...")
lbda = 1
fmin_result = fmin_cg(nn_cost, initial_nn_params, nn_grad, (input_layer_size, hidden_layer_size, num_labels, X, y, lbda), maxiter=50, full_output=True)
nn_params = fmin_result[0].flatten()
Theta1 = np.reshape(nn_params[0 : hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1) : ], (num_labels, hidden_layer_size + 1))

input("Program paused. Press Enter to cotinue.")

# ================= Part 9: Visualize Weights =================

pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {}%'.format(np.mean(pred == y.flatten()) * 100))