import copy
import matplotlib.pyplot as plt


def LU_decompose(A):
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = copy.deepcopy(A)

    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                L[j][i] = U[j][i] / U[i][i]

        for i in range(k, n):
            for j in range(k - 1, n):
                U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j]

    return L, U


def solve_system(L, U, b):
    """
    решает систему: LUx=b
    """
    # Ly = b
    n = len(L)
    y = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) / L[i][i]

    # Ux = y
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(n - 1, i - 1, -1):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    return x


def least_squares(x, y, n):
    """
    Коэффициент подсчета полинома (степень = n) для метода наименьших квадратов для
    аппроксимации табличной функции y = f(x)
    """
    assert len(x) == len(y)
    A = []
    b = []
    for k in range(n + 1):
        A.append([sum(map(lambda x: x ** (i + k), x)) for i in range(n + 1)])
        b.append(sum(map(lambda x: x[0] * x[1] ** k, zip(y, x))))
    L, U = LU_decompose(A)
    return solve_system(L, U, b)


def P(coefs, x):
    """
    Вычисляет значение полиномиальной функции в точке x
    """
    return sum([c * x**i for i, c in enumerate(coefs)])


def sum_squared_errors(x, y, ls_coefs):
    """
    Вычислияет сумму квадратов отклонения
    """
    y_ls = [P(ls_coefs, x_i) for x_i in x]
    return sum((y_i - y_ls_i)**2 for y_i, y_ls_i in zip(y, y_ls))


if __name__ == '__main__':
    x = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    y = [-0.4597, 1.0, 1.5403, 1.5839, 2.010, 3.3464]
    plt.scatter(x, y, color='r')

    print('МНК, degree = 1')
    ls1 = least_squares(x, y, 1)
    print(f'P(x) = {ls1[0]} + {ls1[1]}x')
    plt.plot(x, [P(ls1, x_i) for x_i in x], color='b', label='degree = 1')
    print(f'сумма квадратов отклонения = {sum_squared_errors(x, y, ls1)}')

    print('МНК, degree = 2')
    ls2 = least_squares(x, y, 2)
    print(f'P(x) = {ls2[0]} + {ls2[1]}x + {ls2[2]}x^2')
    plt.plot(x, [P(ls2, x_i) for x_i in x], color='g', label='degree = 2')
    print(f'сумма квадратов отклонения  = {sum_squared_errors(x, y, ls2)}')

    plt.legend()
    plt.show()