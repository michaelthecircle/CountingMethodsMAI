def df(x_test, x, y):
    """
    Вычисляет первую производную табличной функции f(x) = y в точке = x_test
    """
    assert len(x) == len(y)
    for interval in range(len(x)):
        if x[interval] <= x_test < x[interval+1]:
            i = interval
            break
    if (x_test == x[i]) & (i != 0):
        left = (y[i] - y[i - 1]) / (x[i] - x[i - 1])
        right = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        mean = (left + right) / 2
        print('Левая производная: {}'.format(left))
        print('Правая производная: {}'.format(right))
        print('Среднее: {}'.format(mean))
    elif (x_test == x[i + 1]) & (i != len(x) - 1):
        left = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        right = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1])
        mean = (left + right) / 2
        print('Левая производная: {}'.format(left))
        print('Правая производная: {}'.format(right))
        print('Среднее: {}'.format(mean))
    return mean
def SeparatedDiff(X,Y):
    n = len(X)
    if n < 2:
        if n == 1: return Y[0]
        return None
    if n > 2:
        return (SeparatedDiff(X[:n-1], Y[:n-1]) - SeparatedDiff(X[1:], Y[1:]))/(X[0] - X[n-1])
    return (Y[0] - Y[1])/(X[0] - X[1])

# Произведение всех (x-xi),
def BracketsMult(x, X):
    ans = 1
    for i in range(len(X)):
        ans *= (x - X[i])
    return ans
def DerSeveralBrackets(x, X):
    n = len(X)
    if n == 1: return 1 #если скобка одна
    return BracketsMult(x, X[1:]) + BracketsMult(x, X[:1])*DerSeveralBrackets(x, X[1:])

def BracketsMult2(x,X):
    ans = 0
    for i in range(len(X)):
        ans += DerSeveralBrackets(x, X[:i]+X[i+1:])
    return ans

# Нахождение второй производной
def d2f(x, X, Y):
    n = len(X)
    ans = 0
    for i in range(2, n):
        ans += SeparatedDiff(X[:i+1], Y[:i+1])*BracketsMult2(x,X[:i])
    return ans


if __name__ == '__main__':
    x = [0.2, 0.5, 0.8, 1.1, 1.4]
    y = [12.906, 5.5273, 3.8777, 3.2692, 3.0319]
    x_test = 0.8

    print('Первая производная')
    print(f'df({x_test}) = {df(x_test, x, y)}')

    print('Вторая производная')
    print(f'd2f({x_test}) = {d2f(x_test, x, y)}')