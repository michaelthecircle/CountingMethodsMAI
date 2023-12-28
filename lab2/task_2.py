import math
import numpy as np


def f1(X):
    return X[0] ** 2 / 4 + X[1] ** 2 - 1


def f2(X):
    return 2 * X[1] - math.exp(X[0]) - X[0]


def df1_dx1(X):
    # df1 / dx1
    return X[0] / 2


def df1_dx2(X):
    return 2 * X[1]


def df2_dx1(X):
    return -math.exp(X[0]) - 1


def df2_dx2(X):
    return 2


def phi1(X):
    # X1 = phi1(X)
    return 2 * X[1] - math.exp(X[0])


def phi2(X):
    # X2 = phi2(X)
    return math.sqrt(1 - X[0] ** 2 / 4)


def dphi1_dx1(X):
    return -math.exp(X[0])


def dphi1_dx2(X):
    return 2


def dphi2_dx1(X):
    return 1 / (2 * math.sqrt(1 - X[0] ** 2 / 4)) * (-2 * X[0] / 4)


def dphi2_dx2(X):
    return 0


def L_inf_norm(a):
    """
    L_inf норма массива а
    ||a||_inf = max(abs(a))
    """
    abs_a = [abs(i) for i in a]
    return max(abs_a)


def get_q(interval1, interval2):
    """
    q коэфициенты для метода итераций
    """
    l1, r1 = interval1
    l2, r2 = interval2
    m1 = (l1 + r1) / 2
    m2 = (l2 + r2) / 2
    x1 = m1 + abs(r1 - l1)
    x2 = m2 + abs(r2 - l2)
    max1 = abs(dphi1_dx1([x1, x2])) + abs(dphi1_dx2([x1, x2]))
    max2 = abs(dphi2_dx1([x1, x2])) + abs(dphi2_dx2([x1, x2]))
    return max(max1, max2)


def iteration_method(phi1, phi2, intervals, eps, maxIters):
    """
    f1(x1, x2) == 0
    f2(x1, x2) == 0
    x1 = phi1(x1, x2)
    x2 = phi2(x1, x2)
    """
    l1, r1 = intervals[0][0], intervals[0][1]
    l2, r2 = intervals[1][0], intervals[1][1]
    x_prev = [(l1 + r1) * 0.5, (l2 + r2) * 0.5]
    q = get_q(intervals[0], intervals[1])
    iters = 0
    while iters != maxIters:
        iters += 1
        x = [phi1(x_prev), phi2(x_prev)]
        if L_inf_norm([(x[i] - x_prev[i]) for i in range(len(x))]) < eps:
            break
        x_prev = x
    if iters == maxIters:
        print('Достигнут максимум итераций')
    else:
        print('simple iters = ' + str(iters))
        return x, iters


def newton_method(f1, f2, df1_dx1, df1_dx2, df2_dx1, df2_dx2, intervals, eps, maxIters):
    """
    Найти корни системы
    f1(x1, x2) == 0
    f2(x1, x2) == 0
    на интервалах используя метод Ньютона
    """
    l1, r1 = intervals[0][0], intervals[0][1]
    l2, r2 = intervals[1][0], intervals[1][1]
    x_prev = np.array([(l1 + r1) / 2, (l2 + r2) / 2])
    jacobi = [[df1_dx1(x_prev), df1_dx2(x_prev)], [df2_dx1(x_prev), df2_dx2(x_prev)]]
    jacobi_inversed = np.linalg.inv(np.array(jacobi))
    print('матрица Якоби', str(jacobi_inversed), sep='\n')
    iters = 0
    while iters != maxIters:
        iters += 1
        x = x_prev - jacobi_inversed @ np.array([f1(x_prev), f2(x_prev)])
        if L_inf_norm([(x[i] - x_prev[i]) for i in range(len(x))]) < eps:
            break
        x_prev = x
    if iters == maxIters:
        print('Достигнут максимум итераций')
    else:
        print('iters ' + str(iters))
        return x, iters


if __name__ == "__main__":
    with open('/Users/admin/PycharmProjects/CountingMethods/lab2/input_2.txt', 'r') as file:
        l1, r1 = map(float, file.readline().split())
        l2, r2 = map(float, file.readline().split())
        eps = float(file.readline())
        maxIters = int(file.readline())
        print('Метод Ньютона')
        x_newton, i_newton = newton_method(f1, f2, df1_dx1, df1_dx2, df2_dx1, df2_dx2, [(l1, r1), (l2, r2)], eps,
                                           maxIters)
        print('x =', x_newton, '; f1(x) =', f1(x_newton), '; f2(x)=', f2(x_newton))
        print('Итераций:', i_newton)

        print('Метод Итераций')
        x_iter, i_iter = iteration_method(phi1, phi2, [(l1, r1), (l2, r2)], eps, maxIters)
        print('x =', x_iter, '; f1(x) =', f1(x_iter), '; f2(x) =', f2(x_iter))
        print('Итераций:', i_iter)
