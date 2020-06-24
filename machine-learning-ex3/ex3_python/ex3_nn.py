import os
import numpy as np
from scipy.io import loadmat

from sigmoid import sigmoid

# Initialization
os.system('clear')
np.set_printoptions(4, suppress=True)

num_layer_size = 400
hidden_layer_size = 25
num_labels = 10

# =========== Part 1: Loading and Visualizing Data =============

annots = loadmat('ex3data1.mat')
X = annots['X']
y = annots['y']
m = X.shape[0]

# ================ Part 2: Loading Pameters ================

input('Loading Saved Neural Network Parameters ...')

annots = loadmat('ex3weights.mat')
Theta1 = annots['Theta1']
Theta2 = annots['Theta2']

X = np.insert(X, 0, 1, 1)
A = sigmoid(X.dot(Theta1.T))
A = np.insert(A, 0, 1, 1)
B = sigmoid(A.dot(Theta2.T))

pred = np.argmax(B, 1) + 1

print('Train Accuracy: {:.2%}'.format(np.mean(pred == y.flatten())))