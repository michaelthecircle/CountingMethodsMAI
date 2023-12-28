def f(x):
    return x**2 / (x**3 - 27)

####3 4 производную уточнить правую левую границу
def integrate_rectangle_method(f, l, r, h):
    """
    вычисляет интеграл f(x)dx на интервале [l; r] с использованием метода прямоугольника с шагом = h
    """
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
    return result


def integrate_trapeze_method(f, l, r, h):
    """
    вычисляет интеграл f(x)dx на интервале [l; r] с использованием метода трапеций с шагом = h
    """
    result = 0
    cur_x = l
    while cur_x < r:
        result += h * 0.5 * (f(cur_x + h) + f(cur_x))
        cur_x += h
    return result


def integrate_simpson_method(f, l, r, h):
    """
    вычисляет интеграл f(x)dx на интервале [l; r] с использованием метода симпсона с шагом = h
    """
    result = 0
    cur_x = l + h
    while cur_x < r:
        result += f(cur_x - h) + 4*f(cur_x) + f(cur_x + h)
        cur_x += 2 * h
    return result * h / 3


def runge_rombert_method(h1, h2, integral1, integral2, p):
    """
    более точное значение интервала
    работает если h1 == k * h2
    """
    return integral1 + (integral1 - integral2) / ((h2 / h1)**p - 1)


if __name__ == '__main__':
    l, r = -2, 2  # интервал интегрирования
    h1, h2 = 1.0, 0.5  # шаги

    print('Метод прямоугольников')
    int_rectangle_h1 = integrate_rectangle_method(f, l, r, h1)
    int_rectangle_h2 = integrate_rectangle_method(f, l, r, h2)
    print(f'Шаг = {h1}: интеграл = {int_rectangle_h1}')
    print(f'Шаг = {h2}: интеграл = {int_rectangle_h2}')

    print('Метод трапеций')
    int_trapeze_h1 = integrate_trapeze_method(f, l, r, h1)
    int_trapeze_h2 = integrate_trapeze_method(f, l, r, h2)
    print(f'Шаг = {h1}: интеграл = {int_trapeze_h1}')
    print(f'Шаг = {h2}: интеграл = {int_trapeze_h2}')

    print('Метод Симпсона')
    int_simpson_h1 = integrate_simpson_method(f, l, r, h1)
    int_simpson_h2 = integrate_simpson_method(f, l, r, h2)
    print(f'Шаг = {h1}: интеграл = {int_simpson_h1}')
    print(f'Шаг = {h2}: интеграл = {int_simpson_h2}')

    print('метод рунге Ромберга')
    print(f'Более точный интеграл методом прямоугольников = {runge_rombert_method(h1, h2, int_rectangle_h1, int_rectangle_h2, 2)}')
    print(f'Более точный интеграл методом трапеций = {runge_rombert_method(h1, h2, int_trapeze_h1, int_trapeze_h2, 2)}')
    print(f'Более точный интеграл по методу Симпсона = {runge_rombert_method(h1, h2, int_simpson_h1, int_simpson_h2, 4)}')