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
    m = len(y)

    xa = np.append(np.ones((1, m)), np.asmatrix(x), axis=0)
    approx_value = np.sum(np.array(theta) * np.transpose(np.array(xa)), axis=1)
    dif = approx_value - y
    grad = [sum(dif) / m]
    grad.extend(np.sum(dif * x, axis=1) / m)
    return grad


def batch_gradient_descent(data, start_theta=(1, 1)):
    t0 = start_theta[0]
    t1 = start_theta[1]
    x = data[:, 0]
    y = data[:, 1]
    max_iter = 40
    iter_num = 0
    alpha = 0.001

    while iter_num < max_iter:
        gradient = compute_gradient((t0, t1), [x], y)

        temp0 = t0 - alpha * gradient[0]
        temp1 = t1 - alpha * gradient[1]

        t0 = temp0
        t1 = temp1
        iter_num += 1

    return t0, t1

