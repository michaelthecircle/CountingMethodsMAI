import numpy as np
import math
from typing import Sequence

from format.format import print_system


class Matrix():
    def __init__(self, matrix):
        self.m = np.array(matrix, dtype='float64')

    def __str__(self):
        return str(self.m)

    def multiply(self, rhs: 'Matrix') -> 'Matrix':
        """Matrix multipication using numpy."""
        return Matrix(self.m.dot(rhs.m))

    def transpose(self) -> 'Matrix':
        """Transpose a matrix"""
        nrows = len(self.m)
        ncols = len(self.m[0])
        T = [[0] * ncols for _ in range(nrows)]
        for i in range(nrows):
            for j in range(ncols):
                T[j][i] = self.m[i][j]
        return Matrix(T)

    @classmethod
    def identity(self, n: int) -> 'Matrix':
        '''Create an n by n Identity matrix.'''
        I = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
        return Matrix(I)

    @classmethod
    def zeros(self, size: Sequence[int]) -> 'Matrix':
        '''Create and m by n matrix filled with zeros'''
        Z = [[0] * size[0] for _ in range(size[1])]
        return Matrix(Z)

    # 1.1
    def decomposition_LU(self) -> 'list[Matrix, Matrix, Matrix, int]':
        '''
        Perform LU decomposition for a square matrix.
        Returns a list of 3 matrices: [P, L, U, number_of_row_exhanges]
        '''
        U = Matrix(self.m.copy())
        P = Matrix.identity(len(U.m))
        number_of_row_exhanges = 0
        L = Matrix.identity(len(U.m))
        L = Matrix.zeros((len(U.m), len(U.m)))

        # step 1: choose a pivot.
        # take the kth column and find the element with the largest
        # absolute value of all the other elements in this column
        for i in range(len(U.m) - 1):
            max_row_idx = i
            max_elem = U.m[i][i]

            for cur_row in range(i, len(U.m)):
                cur_elem = U.m[cur_row][i]
                if abs(cur_elem) > abs(max_elem):
                    max_elem = cur_elem
                    max_row_idx = cur_row

            # step 2: bring the row with max |element| to the top and perform row swaps
            if max_row_idx != i:
                P.m[[max_row_idx, i]] = P.m[[i, max_row_idx]]
                U.m[[max_row_idx, i]] = U.m[[i, max_row_idx]]
                L.m[[max_row_idx, i]] = L.m[[i, max_row_idx]]
                number_of_row_exhanges += 1

            # step 3: perform row reduction
            for j in range(i + 1, len(U.m)):
                multiplier = U.m[j][i] / max_elem
                U.m[j] -= multiplier * U.m[i]
                L.m[j][i] = multiplier

        # fill the diagonal of L with 1's by adding an Identity matrix to it
        L.m = L.m + Matrix.identity(len(L.m)).m

        return [P, L, U, number_of_row_exhanges]

    def solve_using_LU(self, P_: 'Matrix', L_: 'Matrix', U_: 'Matrix', b: Sequence) -> 'Matrix':
        P = P_.m.copy()
        L = L_.m.copy()
        U = U_.m.copy()

        b_hat = P.dot(b)
        # now solving LUx = b_hat, where Ux=z

        # first solve Ly = b_hat
        y = b_hat.copy()
        for i in range(0, len(L) - 1):
            for cur_row in range(i + 1, len(L)):
                coeff = L[cur_row][i] / L[i][i]
                y[cur_row] -= y[i] * coeff

        # now solve Ux=y
        x = y.copy()

        for i in reversed(range(0, len(U))):
            for cur_row in range(0, i):
                coeff = U[cur_row][i] / U[i][i]
                x[cur_row] -= x[i] * coeff

        for i in range(len(x)):
            x[i] /= U[i][i]

        return Matrix(x)

    def inverse_matrix_using_LU(self) -> 'Matrix':
        '''Get the inverse of a matrix using LU decomposition'''
        P, L, U, _ = self.decomposition_LU()
        inverse = Matrix.zeros((len(self.m), len(self.m)))

        # Compute the inverse matrix column by column
        # LUx = b, where b is a column of an Identity matrix
        for i, b in enumerate(P.transpose().m):
            # Ld = b
            d = L.solve_using_LU(Matrix.identity(len(L.m)), L, Matrix.identity(len(L.m)), b)
            # Ux = d
            x = U.solve_using_LU(Matrix.identity(len(U.m)), Matrix.identity(len(U.m)), U, d.m)
            # x becomes an ith column of the inverse matrix
            inverse.m[i] = x.m

        return inverse.transpose()

    def det(self) -> float:
        '''Compute the determinant of a matrix using LU decomposition'''
        U, number_of_row_exhanges = self.decomposition_LU()[2:]
        result = 1.0

        # determinant = product of all the diagonal entries of U
        # It reverses sign in case we perform an odd number of permutations
        for i in range(len(U.m)):
            result *= U.m[i][i]
        if number_of_row_exhanges % 2 != 0:
            result *= -1

        return result
    #ДЛЯ ВТОРОГО ЗАДАНИЯ
    def tridiagonal_matrix_algorithm(self, b: Sequence) -> 'Matrix':
        '''Returns a solution vector x to Ax=b, where A is a tridiagonal matrix.'''
        b = b.copy()
        A = self.m.copy()
        x = Matrix.zeros((len(b), 1)).m[0]
        # Forward substitution
        for row_idx in range(1, len(self.m)):
            proportion = A[row_idx, row_idx - 1] / A[row_idx - 1, row_idx - 1]
            A[row_idx] -= A[row_idx - 1] * proportion
            b[row_idx] -= b[row_idx - 1] * proportion

        x[len(x) - 1] = b[len(x) - 1] / A[len(x) - 1, len(x) - 1]
        for i in reversed(range(len(x) - 1)):
            x[i] = (b[i] - A[i, i + 1] * x[i + 1]) / A[i, i]

        return Matrix(x)

    # ДЛЯ ТРЕТЬЕГО ЗАДАНИЯ
    def l1_norm(self) -> float:
        """Return l1 norm of a matrix or a vector"""
        norm = 0
        if self.m.ndim == 2:
            for col_idx in range(len(self.m[0])):
                s = 0
                for row_idx in range(len(self.m)):
                    s += abs(self.m[row_idx, col_idx])
                if s > norm:
                    norm = s
            return norm
        ret = sum([abs(i) for i in self.m])
        return ret

    def simple_iteration_method(self, b: Sequence, eps=0.01) -> 'list[Matrix, int]':
        """Solve a system of linear equations using the simple iteration method"""
        A = self.m.copy()

        # Create alpha and beta arrays
        beta = np.array([b[i] / A[i, i] for i in range(len(b))])
        alpha = []
        for i in range(len(A)):
            alpha.append([-A[i, j] / A[i, i] if i != j else 0 for j in range(len(A[0]))])
        alpha = Matrix(alpha)

        # Perform iterations
        x = beta.copy()
        iter_count = 0
        eps_k = 0

        use_norm = alpha.l1_norm() < 1
        while iter_count == 0 or eps_k > eps:
            iter_count += 1
            x_prev = x.copy()
            x = beta + alpha.m.dot(x_prev)

            if use_norm:
                eps_k = alpha.l1_norm() / (1 - alpha.l1_norm()) * Matrix(x - x_prev).l1_norm()
            else:
                eps_k = (x - x_prev).l1_norm()

        return x, iter_count

    def Seidel_method(self, b: Sequence, eps=0.01) -> 'list[Matrix, int]':
        """Solve a system of linear equations using Seidel method"""
        A = self.m.copy()

        beta = np.array([b[i] / A[i, i] for i in range(len(b))])
        alpha = []
        for i in range(len(A)):
            alpha.append([-A[i, j] / A[i, i] if i != j else 0 for j in range(len(A[0]))])
        alpha = Matrix(alpha)

        c, d = Matrix.zeros((len(A), len(A[0]))), Matrix.zeros((len(A), len(A[0])))
        for i in range(len(A)):
            for j in range(len(A[0])):
                if j >= i:
                    d.m[i, j] = alpha.m[i, j]
                else:
                    c.m[i, j] = alpha.m[i, j]

        inverse = Matrix(Matrix.identity(len(A)).m - c.m).inverse_matrix_using_LU()
        alpha = inverse.multiply(d)
        beta = inverse.multiply(Matrix(beta))
        x = beta.m.copy()
        iter_count = 0
        eps_k = 0
        use_norm = alpha.l1_norm() < 1

        while iter_count == 0 or eps_k > eps:
            iter_count += 1
            x_prev = x.copy()
            x = beta.m + alpha.m.dot(x_prev)

            if use_norm:
                eps_k = alpha.l1_norm() / (1 - alpha.l1_norm()) * Matrix(x - x_prev).l1_norm()
            else:
                eps_k = (x - x_prev).l1_norm()

        return x, iter_count