import math


def f(x):
    return math.log(x + 1) - 2 * x + 0.5  # ln(x+1) - 2x + 0.5


def phi(x):
    return (math.log(x + 1) + 0.5) / 2


def df(x):
    return 1.0 / (x + 1) - 2


def iteration_method(f, phi, interval, eps, maxIters):
    """
    Найти корень при f(x) == 0 использую метод итераций
    Возвращает x и количество итераций
    """
    l, r = interval[0], interval[1]
    x_prev = (l + r) * 0.5
    iters = 0
    while iters != maxIters:
        iters += 1
        x = phi(x_prev)
        if abs(f(x) - f(x_prev)) < eps:
            break
        x_prev = x
    if (iters == maxIters):
        print('Достигнут максимум итераций')
    else:
        return x, iters


def newton_method(f, df, interval, eps, maxIters):
    """
    Найти корень при f(x) == 0 используя метод ньютона
    Возвращает x и количество итераций
    """
    l, r = interval[0], interval[1]
    x_prev = (l + r) * 0.5
    iters = 0
    while iters != maxIters:
        iters += 1
        x = x_prev - f(x_prev) / df(x_prev)
        if abs(f(x) - f(x_prev)) < eps:
            break
        x_prev = x
    if iters == maxIters:
        print('Достигнут максимум итераций')
    else:
        return x, iters


if __name__ == "__main__":
    with open('/Users/admin/PycharmProjects/CountingMethods/lab2/input_1.txt', 'r') as file:
        # Ввод интервала и эпсилона
        l = int(file.readline())
        r = int(file.readline())
        eps = float(file.readline())
        maxIters = int(file.readline())
        print('Метод итераций')
        x_iter, i_iter = iteration_method(f, phi, (l, r), eps, maxIters)
        print('x =', x_iter, '; f(x) =', f(x_iter))
        print('итераций:', i_iter)

        print('Метод Ньютона')
        x_newton, i_newton = newton_method(f, df, (l, r), eps, maxIters)
        print('x =', x_newton, '; f(x) =', f(x_newton))
        print('итераций:', i_newton)
