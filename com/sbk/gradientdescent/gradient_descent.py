import numpy as np


def step(v, direction, learning_rate):
    return v - direction * learning_rate


def is_points_close_enough(cur_x, next_x, tolerance):
    return np.abs(next_x - cur_x) > tolerance


def gradient_descent_single_variable(func_calc_gradient, func, start_x=0, tolerance=0.000001, learning_rate=0.01):
    cur_x = start_x

    is_not_converged = True

    while is_not_converged:
        cur_gradient = func_calc_gradient(cur_x)
        next_x = step(cur_x, cur_gradient, learning_rate)
        is_not_converged = is_points_close_enough(cur_x, next_x, tolerance)
        cur_x = next_x

    return cur_x, func(cur_x)
