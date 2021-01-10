import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from plot_data import plot_data
from linear_reg_cost_function import linear_reg_cost_function

# Initialization
os.system('clear')
np.set_printoptions(4, suppress=True)

# =========== Part 1: Loading and Visualizing Data =============

annots = loadmat('ex5data1.mat')
X = annots['X']
y = annots['y']
Xval = annots['Xval']
yval = annots['yval']
Xtest = annots['Xtest']
ytest = annots['ytest']
m = X.shape[0]

data = pd.DataFrame()
data['X'] = X.flatten()
data['y'] = y.flatten()
data = data.sort_values('X')
plot_data(data)


# ================ Part 2: Loading Pameters ================

theta = np.array([[1],[1]])
J = linear_reg_cost_function(np.insert(X, 0, values=1, axis=1), y, theta, 1)
print("cost at theta = [1, 1] is {}, (this value should be about 303.993192)".format(J))
input("Program paused. Press enter to continue.")
