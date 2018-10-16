import numpy as np


def func(t):
    return 5*(t-3)**4 + 2*t - 12


def calc_gradient(t):
    return 20*(t-3)**3 + 2


def step(v, direction, step_size):
    return v + direction*step_size


def gradient_descent(func_calc_gradient, start_x = 0, tolerance=0.001, learning_rate = -0.001):

    x = start_x

    while True:
        gd = func_calc_gradient(x) # compute the gradient at v
        next_x = step(x, gd, learning_rate) # take a negative gradient step
        if np.abs(next_x - x) < tolerance: # stop if we're converging
            break
        x = next_x

    return x, func(x)