import numpy as np

def sign(x):
    return -1 if x < 0 else 1 if x > 0 else 0


def L2_norm(vec):
    ans = 0
    for num in vec:
        ans += num * num
    return np.sqrt(ans)


def get_householder_matrix(A, col_num):
    """
    Получение матрицы Хаусхолдера для итераций QR разложения
    """
    n = A.shape[0]
    v = np.zeros(n)
    a = A[:, col_num]
    v[col_num] = a[col_num] + sign(a[col_num]) * L2_norm(a[col_num:])
    for i in range(col_num + 1, n):
        v[i] = a[i]
    v = v[:, np.newaxis]
    H = np.eye(n) - (2 / (v.T @ v)) * (v @ v.T)
    return H


def QR_decomposition(A):
    """
    делает QR разложение: A = QR
    возвращает Q, R
    """
    n = A.shape[0]
    Q = np.eye(n)
    A_i = np.copy(A)

    for i in range(n - 1):
        H = get_householder_matrix(A_i, i)
        Q = Q @ H
        A_i = H @ A_i
    return Q, A_i


def get_roots(A, i):
    """
    Получение корней системы из 2х уравнений (i and i+1) матрицы A
    """
    n = A.shape[0]
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0
    a21 = A[i + 1][i] if i + 1 < n else 0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


def is_complex(A, i, eps):
    """
    Проверка является ли  i и (i+1)-ый собственные значения комплексные
    """
    Q, R = QR_decomposition(A)
    A_next = np.dot(R, Q)
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return abs(lambda1[0] - lambda2[0]) <= eps and abs(lambda1[1] - lambda2[1]) <= eps


def get_eigen_value(A, i, eps):
    """
    Получает i-th (и (i+1)-th если комплекс.) собственное значение матрицы А
    """
    A_i = np.copy(A)
    iters = 0
    while True:
        iters += 1
        Q, R = QR_decomposition(A_i)
        A_i = R @ Q
        print('            матрица \n' + str(A_i))
        if L2_norm(A_i[i + 1:, i]) <= eps:
            return A_i[i][i], A_i
        elif L2_norm(A_i[i + 2:, i]) <= eps and is_complex(A_i, i, eps):
            print('iters when complex = ' + str(iters))
            return get_roots(A_i, i), A_i
def get_eigen_values_QR(A, eps):
    """
    Получает все собственные знаечния А используя QR разложение
    """
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_values = []

    i = 0
    while i < n:
        cur_eigen_values, A_i_plus_1 = get_eigen_value(A_i, i, eps)
        if isinstance(cur_eigen_values, np.ndarray):
            # комплексные
            eigen_values.extend(cur_eigen_values)
            i += 2
        else:
            # вещественные
            eigen_values.append(cur_eigen_values)
            i += 1
        A_i = A_i_plus_1
    return eigen_values


if __name__ == '__main__':
    A = []
    with open('/Users/admin/PycharmProjects/CountingMethods/lab1/input_5.txt', 'r') as file:
        n = int(file.readline())
        eps = float(file.readline())
        for line in file:
            row = list(map(int, line.split()))
            A.append(row)
    A = np.array(A, dtype='float')

    eig_values = get_eigen_values_QR(A, eps)
    print('Собственные значения:', eig_values)