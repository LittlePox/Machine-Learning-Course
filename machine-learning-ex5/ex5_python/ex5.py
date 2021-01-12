import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
from plot_data import plot_data
from linear_reg_cost_function import cost_function
from linear_reg_cost_gradient import cost_gradient
from train_linear_reg import train_linear_reg
from learning_curve import learning_curve
from poly_features import poly_features

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
J = cost_function(np.insert(X, 0, values=1, axis=1), y, theta, 1)
print("cost at theta = [1, 1] is {}, (this value should be about 303.993192)".format(J))
input("Program paused. Press enter to continue.")

# =========== Part 3: Regularized Linear Regression Gradient =============

grad = cost_gradient(np.insert(X, 0, values=1, axis=1), y, theta, 1)
print("gradient at theta = [1, 1] is {}, (this value should be about [-15.303016; 598.250744])".format(grad))
input("Program paused. Press enter to continue.")

# =========== Part 4: Train Linear Regression =============

lbda = 0
theta = train_linear_reg(np.insert(X, 0, values=1, axis=1), y, lbda)
_X = np.sort(X)
_X = np.insert(_X, 0, values=1, axis=1)
plt.plot(np.sort(X), _X.dot(theta), 'r-')
plt.show(block=False)
plt.close()
input("Program paused. Press enter to continue.")

# =========== Part 5: Learning Curve for Linear Regression =============

(error_train, error_test) = learning_curve(np.insert(X, 0, values=1, axis=1), y, np.insert(Xval, 0, values=1, axis=1), yval, lbda)

plt.plot(range(1, m + 1), error_train, range(1, m + 1), error_test)
plt.title('Learning curve for linear regression')
plt.legend(('Train', 'Cross Validation'))
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show(block=False)

print("# Training Examples\tTrain Error\tCross Validation Error")
for i in range(0, m):
    print("\t{}\t\t{}\t{}".format(i, error_train[i][0], error_test[i][0]))
input("Program paused. Press enter to continue.")

# =========== Part 6: Feature Mapping for Polynomial Regression =============

X_poly = poly_features
