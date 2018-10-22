import numpy as np


def step(v, direction, learning_rate):
    return v - direction * np.asarray(learning_rate)


def is_points_not_close_enough(cur_x, next_x, tolerance):
    a_cur_x = np.asarray(cur_x)
    a_next_x = np.asarray(next_x)
    distance = np.asarray(a_cur_x - a_next_x)
    distance_s = np.abs(np.sum(distance))
    return distance_s > tolerance


def gradient_descent_single_variable(func_calc_gradient, func, start_x=0, tolerance=0.000001, learning_rate=0.01):
    cur_x = start_x

    convergence_path = []

    is_not_converged = True

    while is_not_converged:
        cur_gradient = func_calc_gradient(cur_x)
        next_x = step(cur_x, cur_gradient, learning_rate)
        is_not_converged = is_points_not_close_enough(cur_x, next_x, tolerance)
        convergence_path.append(next_x)
        cur_x = next_x

    return cur_x, func(cur_x), convergence_path


def gradient_descent_multiple_variable(func_calc_gradient, func, start_x=(3, 5), tolerance=0.00001,
                                       learning_rate=(0.01, 0.01)):
    cur_x = start_x
    convergence_path = []
    is_not_converged = True

    while is_not_converged:
        cur_gradient = func_calc_gradient(cur_x)
        next_x = step(cur_x, cur_gradient, learning_rate)
        is_not_converged = is_points_not_close_enough(cur_x, next_x, tolerance)
        convergence_path.append(next_x)
        cur_x = next_x
    return cur_x, func(cur_x), convergence_path
