import numpy as np
from train_linear_reg import train_linear_reg
from linear_reg_cost_function import cost_function

def learning_curve(X, y, Xval, yval, lbda):
    m = X.shape[0]
    error_train = np.zeros((m, 1))
    error_test = np.zeros((m, 1))
    for i in range(0, m):
        Xtrain = X[0:i+1,]
        ytrain = y[0:i+1,]
        theta = train_linear_reg(Xtrain, ytrain, lbda)
        error_train[i] = cost_function(Xtrain, ytrain, theta, lbda)
        error_test[i] = cost_function(Xval, yval, theta, lbda)
    return (error_train, error_test)