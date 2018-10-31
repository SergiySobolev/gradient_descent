import numpy as np


def mccormik_function(t):
    x = t[0]
    y = t[1]
    return np.sin(x + y) \
           + np.square(x - y) \
           - 1.5 * x \
           + 2.5 * y \
           + 1


def calc_gradient_mccormik_function(t):
    x = t[0]
    y = t[1]
    return [np.cos(x + y) + 2 * (x - y) - 1.5, np.cos(x + y) - 2 * (x - y) + 2.5]
