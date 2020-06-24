from sigmoid import sigmoid

def sigmoid_gradient(X):
    t = sigmoid(X)
    return t * (1 - t)
