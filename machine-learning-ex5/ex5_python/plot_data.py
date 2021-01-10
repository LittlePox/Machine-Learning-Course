import matplotlib.pyplot as plt

def plot_data(data):
    graph = data.plot(kind = 'scatter', x=0, y=1, c='r', marker='x', s=10)
    graph.set_xlabel('Change in water level (x)')
    graph.set_ylabel('Water flowing out of the dam (y)')
    plt.show(block=False)
