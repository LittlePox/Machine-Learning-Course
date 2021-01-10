import numpy as np
from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    h = np.insert(X, 0, 1, 1)
    h = sigmoid(h.dot(Theta1.T))
    h = np.insert(h, 0, 1, 1)
    h = sigmoid(h.dot(Theta2.T))
    return np.argmax(h, 1) + 1