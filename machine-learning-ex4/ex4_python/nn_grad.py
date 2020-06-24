import numpy as np
from sigmoid import sigmoid
from sigmoid_gradient import sigmoid_gradient

def nn_grad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lbda):
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

    delta3 = (y_bar - Y).T # 10 * 5000
    delta2 = Theta2.T.dot(delta3) * sigmoid_gradient(np.insert(X.dot(Theta1.T), 0, 1, 1).T) # 26 * 5000
    delta2 = delta2[1:,:] # 25 * 500
    Delta2 = delta3.dot(h1) # 10 * 26
    Delta1 = delta2.dot(X) # 25 × 401

    Theta1_grad = (Delta1 + lbda * theta1) / m
    Theta2_grad = (Delta2 + lbda * theta2) / m

    return np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
