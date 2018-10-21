def func_1(t):
    return 5 * (t - 3) ** 4 + 2 * t - 12


def calc_gradient_1(t):
    return 20 * (t - 3) ** 3 + 2


def func_2(t1, t2):
    return (t1 - 5) ** 2 + (t2 - 7) ** 2


def calc_gradient_2(t):
    return 20 * (t - 3) ** 3 + 2
