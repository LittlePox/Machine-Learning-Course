import numpy as np
from sigmoid import sigmoid

def nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbda):
    Theta1 = np.reshape(nn_params[0 : hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1) : ], (num_labels, hidden_layer_size + 1))
    m = X.shape[0]
    X = np.insert(X, 0, 1, 1)
    h1 = sigmoid(X.dot(Theta1.T))
    h1 = np.insert(h1, 0, 1, 1)
    y_bar = sigmoid(h1.dot(Theta2.T))
    Y = np.zeros((m, num_labels))
    for i in range(0, m):
        Y[i][y[i] - 1] = 1
    sum = 0
    for i in range(0, m):
        a = Y[i].reshape((1, num_labels))
        b = y_bar[i].reshape((num_labels, 1))
        diff = (a - 1).dot(np.log(1 - b)) - a.dot(np.log(b))
        diff = diff[0][0]
        sum += diff
    theta1 = np.array(Theta1, copy=True)
    theta1[:,0] = 0
    theta2 = np.array(Theta2, copy=True)
    theta2[:,0] = 0
    theta = np.concatenate([theta1.flatten(), theta2.flatten()]).flatten()
    return (sum + 0.5 * lbda * np.inner(theta, theta)) / m

