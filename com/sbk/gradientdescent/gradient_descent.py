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


def compute_cost_function(theta, x, y):
    m = len(y)
    xa = np.append(np.ones((1, m)), np.asmatrix(x), axis=0)
    approx_value = np.sum(np.array(theta) * np.transpose(np.array(xa)), axis=1)
    dif = approx_value - y
    return sum(dif ** 2) / (2 * m)


def compute_gradient(theta, x, y):
    x_arr = np.asarray(x) if type(x) is list else x
    m = x_arr.shape[0]
    ones = np.ones((m, 1))
    xa = np.hstack((ones, x))
    approx_value = np.sum(theta * xa, axis=1)
    dif = approx_value - y
    grad0 = np.asarray([sum(dif) / m])
    gradn = np.dot(np.asarray(dif), x) / m
    grad = np.concatenate((grad0, gradn), axis=0)
    return grad


def propose_theta(data):
    return np.ones(data.shape[1])


def update_theta(theta, gradient, alpha):
    return theta - alpha * gradient


def batch_gradient_descent(data, start_theta=None, alpha=None, max_iter=None):
    if start_theta is None:
        start_theta = propose_theta(data)

    if alpha is None:
        alpha = 0.001

    if max_iter is None:
        max_iter = 40

    f_n = data.shape[1] - 1
    x = data[:, range(f_n)]
    y = data[:, f_n]
    iter_num = 0
    cur_theta = start_theta

    while iter_num < max_iter:
        gradient = compute_gradient(cur_theta, x, y)

        cur_theta = update_theta(cur_theta, gradient, alpha)

        iter_num += 1

    return cur_theta
