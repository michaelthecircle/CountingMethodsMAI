import numpy as np

from format.format import print_system
from lab1.Matrix import Matrix

if __name__ == '__main__':
    #==========ЗАДАНИЕ 1==========#
    with open('/Users/admin/PycharmProjects/CountingMethods/lab1/Solution_1.txt', 'w') as outputFile:
        with open('/Users/admin/PycharmProjects/CountingMethods/lab1/input_1.txt', 'r') as file:
            lines = file.readlines()
            aInput = []
            bInput = []
            for line in lines:
                parts = line.split('=')
                coefficients = list(map(float, parts[0].split()))
                result = float(parts[1])
                aInput.append(coefficients)
                bInput.append(result)

        A = Matrix(aInput)

        np.set_printoptions(suppress=True)

        P, L, U, n_swaps = A.decomposition_LU()
        outputFile.write('        Исходная матрица:\n')
        for i in print_system(aInput, bInput):
            outputFile.write(str(i))
        outputFile.write('        Решение системы:\n')
        outputFile.write(f'Ax = b: {A.solve_using_LU(P, L, U, bInput)}\n')

        outputFile.write(f'\nматрица P:\n{P}\n')
        outputFile.write(f'\nматрица L:\n{L}\n')
        outputFile.write(f'\nматрица U:\n{U}\n')

        outputFile.write(f'        Обратная матрица A:\n{A.inverse_matrix_using_LU()}\n')
        outputFile.write(f'        Определитель матрицы A: {A.det()}')
        outputFile.write('\n' + str(A.multiply(A.inverse_matrix_using_LU())))
    outputFile.close()
    file.close()
    # ==========ЗАДАНИЕ 2==========#
    a = []
    b = []
    with open('/Users/admin/PycharmProjects/CountingMethods/lab1/Solution_2.txt', 'w') as outputFile:
        with open('/Users/admin/PycharmProjects/CountingMethods/lab1/input_2.txt', 'r') as test_file:
            n = int(test_file.readline())
            a.append([int(i) for i in test_file.readline().split()] + [0] * (n - 2))
            for i in range(0, n - 1):
                vec = [0] * i + [float(num) for num in test_file.readline().split()] + [0] * (n - 3 - i)
                a.append(vec)
            b = [int(num) for num in test_file.readline().split()]

        A = Matrix(a)
        outputFile.write('  Матрица А \n' + str(A))
        outputFile.write('\n')
        np.set_printoptions(suppress=True)

        x = A.tridiagonal_matrix_algorithm(b)
        outputFile.write(f'Решение Ax = b: {x}\n')

        outputFile.write(f'Ax = {A.multiply(x)}\n')
        outputFile.write(f'b = {b}')
    outputFile.close()
    test_file.close()
    # ==========ЗАДАНИЕ 3==========#
    a = []
    b = []
    eps = 0.
    with open('/Users/admin/PycharmProjects/CountingMethods/lab1/Solution_3.txt', 'w') as outputFile:
        with open('/Users/admin/PycharmProjects/CountingMethods/lab1/input_3.txt', 'r') as test_file:
            n, eps = [float(i) for i in test_file.readline().split()]
            for _ in range(int(n)):
                vec = [float(num) for num in test_file.readline().split()]
                a.append(vec)
            b = [int(num) for num in test_file.readline().split()]

        A = Matrix(a)

        np.set_printoptions(suppress=True)

        result_iter = A.simple_iteration_method(b, eps)
        outputFile.write('Метод простых итераций:\n')
        outputFile.write(f'x = {result_iter[0]} with epsilon = {eps} после {result_iter[1]} итераций\n')

        result_Seidel = A.Seidel_method(b, eps)
        outputFile.write( '\nМетод Зейделя:\n')
        outputFile.write(f'x = {result_Seidel[0]} with epsilon = {eps} после {result_Seidel[1]} итераций\n')
    outputFile.close()
    test_file.close()
