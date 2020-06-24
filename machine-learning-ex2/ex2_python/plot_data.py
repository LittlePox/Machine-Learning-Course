import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    plt.plot(X[pos][:,0], X[pos][:,1], 'k+', lw=2, ms=7)
    plt.plot(X[neg][:,0], X[neg][:,1], 'ko', mfc='y', ms=7)
    plt.show(block=False)
