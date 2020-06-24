import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(theta, X, y, map_feature, low=30, high=100, num=71):
    u = np.linspace(low, high, num)
    v = np.linspace(low, high, num)
    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i][j] = map_feature(np.array([[u[i], v[j]]])).dot(theta)[0][0]
    plt.contour(u, v, z.T, (0,), linewidths=2)
    plt.show(block=False)