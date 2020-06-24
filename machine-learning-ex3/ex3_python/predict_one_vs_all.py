import numpy as np
from sigmoid import sigmoid

def predict_one_vs_all(all_theta, X):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    p = np.zeros((m, 1))
    X = np.insert(X, 0, 1, 1)

    prob = sigmoid(X.dot(all_theta.T))
    return np.argmax(prob, 1) + 1