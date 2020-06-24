import math

def display_data(X, example_width = None):
    m, n = X.shape
    if example_width is None:
        example_width = math.sqrt(n)
    example_height = n / example_width

    