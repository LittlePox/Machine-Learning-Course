import numpy as np
from debug_initialize_weights import debug_initialize_weights
from nn_grad import nn_grad
from nn_cost import nn_cost
from compute_numerical_gradient import compute_numerical_gradient

def check_nn_gradients(lbda = 0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    Theta1 = debug_initialize_weights(hidden_layer_size, input_layer_size)
    Theta2 = debug_initialize_weights(num_labels, hidden_layer_size)

    X = debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(range(1, m+1), num_labels)

    nn_params = np.concatenate((Theta1.flatten(), Theta2.flatten()))

    def cost_func(p):
        return nn_cost(p, input_layer_size, hidden_layer_size, num_labels, X, y, lbda)

    grad = nn_grad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbda)
    num_grad = compute_numerical_gradient(cost_func, nn_params)

    print((grad, num_grad))
    print('The above two columns you get should be very similar.\n (Left-Your Numerical Gradient, Right-Analytical Gradient)')

    diff = np.linalg.norm(num_grad - grad) / np.linalg.norm(num_grad + grad)

    print('If your backpropagation implementation is correct, then the relative difference will be small (less than 1e-9): ',diff)
