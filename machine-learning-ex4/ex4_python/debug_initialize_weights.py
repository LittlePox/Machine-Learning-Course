import numpy as np

def debug_initialize_weights(fan_out, fan_in):
    count = fan_out * (1 + fan_in)
    return np.reshape(np.sin(range(1, count+1)), (fan_in + 1, fan_out)).T / 10
