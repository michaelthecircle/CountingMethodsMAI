def df(x_test, x, y):
    """
    Вычисляет первую производную табличной функции f(x) = y в точке = x_test
    """
    assert len(x) == len(y)
    for interval in range(len(x)):
        if x[interval] <= x_test < x[interval+1]:
            i = interval
            break

    a1 = (y[i+1] - y[i]) / (x[i+1] - x[i])
    a2 = ((y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - a1) / (x[i+2] - x[i]) * (2*x_test - x[i] - x[i+1])
    return a1 + a2


def d2f(x_test, x, y):
    """
    Вычисляет вторую производную табличной функции f(x) = y в точке = x_test
    """
    assert len(x) == len(y)
    for interval in range(len(x)):
        if x[interval] <= x_test < x[interval+1]:
            i = interval
            break

    num = (y[i+2] - y[i+1]) / (x[i+2] - x[i+1]) - (y[i+1] - y[i]) / (x[i+1] - x[i])
    return 2 * num / (x[i+2] - x[i])


if __name__ == '__main__':
    x = [0.2, 0.5, 0.8, 1.1, 1.4]
    y = [12.906, 5.5273, 3.8777, 3.2692, 3.0319]
    x_test = 0.8

    print('Первая производная')
    print(f'df({x_test}) = {df(x_test, x, y)}')

    print('Вторая производная')
    print(f'd2f({x_test}) = {d2f(x_test, x, y)}')