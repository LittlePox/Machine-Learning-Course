import matplotlib.pyplot as plt

def plot_data(data):
    graph = data.plot(kind = 'scatter', x=0, y=1, c='r', marker='x', s=10)
    graph.set_xlabel('Population of City in 10,000s')
    graph.set_ylabel('Profit in 10,000s')
    plt.show(block=False)
