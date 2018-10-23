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


def compute_cost_function(m, t0, t1, x, y):
    return 1 / 2 / m * sum([(t0 + t1 * np.asarray([x[i]]) - y[i]) ** 2 for i in range(m)])


def batch_gradient_descent(data, start_theta=(1, 1)):
    t0 = start_theta[0]
    t1 = start_theta[1]
    x = data[:, 0]
    y = data[:, 1]
    m = len(x)
    max_iter = 40
    iter_num = 0
    alpha = 0.001

    while iter_num < max_iter:
        grad0 = calc_grad0(m, t0, t1, x, y)
        grad1 = calc_grad1(m, t0, t1, x, y)

        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1

        t0 = temp0
        t1 = temp1
        iter_num += 1

    return t0, t1


def calc_grad1(m, t0, t1, x, y):
    cur_values = t0 + t1 * x
    dif = cur_values - y
    v = dif * x
    return sum(v) / m


def calc_grad0(m, t0, t1, x, y):
    cur_values = t0 + t1 * x
    dif = cur_values - y
    return sum(dif) / m
