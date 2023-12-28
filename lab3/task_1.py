import math


def f(x):
    return math.cos(x) + x


def lagrange_interpolation(x, y, test_point):

    assert len(x) == len(y)
    polynom_str = 'L(x) ='
    polynom_test_value = 0  # L(x*)
    for i in range(len(x)):
        cur_enum_str = ''  # текущий член полинома
        cur_enum_test = 1  # текущее значения полинома для теста
        cur_denom = 1
        for j in range(len(x)):
            if i == j:
                continue
            cur_enum_str += f'(x-{x[j]:.2f})'
            cur_enum_test *= (test_point[0] - x[j])
            cur_denom *= (x[i] - x[j])

        polynom_str += f' + {(y[i] / cur_denom):.2f}*' + cur_enum_str
        polynom_test_value += y[i] * cur_enum_test / cur_denom
    print("значение в точке 1.0 " + str(polynom_test_value))
    return polynom_str, abs(polynom_test_value - test_point[1])


def newton_interpolation(x, y, test_point):
    """
    x - массив координат x
    y - массив координат y
    test_point = (x*, y*) - точка проверки ошибки интерполяции
    """
    assert len(x) == len(y)

    # поиск коэфициентов
    n = len(x)
    coefs = [y[i] for i in range(n)]
    for i in range(1, n):
        for j in range(n - 1, i - 1, -1):
            coefs[j] = float(coefs[j] - coefs[j - 1]) / float(x[j] - x[j - i])

    # получаем полином
    polynom_str = 'P(x) = '
    polynom_test_value = 0  # P(x*)

    cur_multipliers_str = ''
    cur_multipliers = 1
    for i in range(n):
        polynom_test_value += cur_multipliers * coefs[i]
        if i == 0:
            polynom_str += f'{coefs[i]:.2f}'
        else:
            polynom_str += ' + ' + cur_multipliers_str + '*' + f'{coefs[i]:.2f}'

        cur_multipliers *= (test_point[0] - x[i])
        cur_multipliers_str += f'(x-{x[i]:.2f})'
    print("значение в точке 1.0 " + str(polynom_test_value))
    return polynom_str, abs(polynom_test_value - test_point[1])


if __name__ == '__main__':
    x_a = [0, math.pi / 6, 2 * math.pi / 6, 3 * math.pi / 6]
    x_b = [0, math.pi / 6, math.pi / 4, math.pi / 2]
    y_a = [f(x) for x in x_a]
    y_b = [f(x) for x in x_b]

    x_test = 1
    y_test = f(x_test)

    print('Интерполяция Лагранжа')
    print('точки A')
    lagrange_polynom_a, lagrange_error_a = lagrange_interpolation(x_a, y_a, (x_test, y_test))
    print('многочлен:')
    print(lagrange_polynom_a)
    print('модуль ошибки =', lagrange_error_a)

    print('точки B')
    lagrange_polynom_b, lagrange_error_b = lagrange_interpolation(x_b, y_b, (x_test, y_test))
    print('многочлен: ')
    print(lagrange_polynom_b)
    print('модуль ошибки =', lagrange_error_b)
    print()

    print('Интерполяция Ньютона')
    print('точки A')
    newton_polynom_a, newton_error_a = newton_interpolation(x_a, y_a, (x_test, y_test))
    print('многочлен:')
    print(newton_polynom_a)
    print('модуль ошибки =', newton_error_a)

    print('точки B')
    newton_polynom_b, newton_error_b = newton_interpolation(x_b, y_b, (x_test, y_test))
    print('многочлен:')
    print(newton_polynom_b)
    print('модуль ошибки =', newton_error_b)