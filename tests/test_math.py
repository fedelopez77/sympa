"""
Explicit numeric results are calculated in according to:
https://www.symbolab.com/solver/complex-numbers-calculator/
"""

import torch
import unittest
import sympa.math.symmetric_math as sm
import sympa.tests
from tests.utils import get_random_symmetric_matrices


class TestBasicMath(sympa.tests.TestCase):

    def test_to_symmetric(self):
        x = get_random_symmetric_matrices(10, 4)
        x_real = sm.real(x)
        x_imag = sm.imag(x)

        x_real_transpose = x_real.transpose(-1, -2)
        x_imag_transpose = x_imag.transpose(-1, -2)

        self.assertAllEqual(x_real, x_real_transpose)
        self.assertAllEqual(x_imag, x_imag_transpose)

    def test_stick(self):
        x_real = torch.rand(10, 2, 4, 4)
        x_imag = torch.rand(10, 2, 4, 4)

        x = sm.stick(x_real, x_imag)

        self.assertAllEqual(x_real, sm.real(x))
        self.assertAllEqual(x_imag, sm.imag(x))

    def test_conjugate(self):
        x = get_random_symmetric_matrices(10, 4)
        x_imag = sm.imag(x)

        conj_x = sm.conjugate(x)

        self.assertAllEqual(-x_imag, sm.imag(conj_x))
        self.assertAllEqual(x_imag, sm.imag(x))

    def test_transpose(self):
        x = get_random_symmetric_matrices(10, 4)
        x_real = sm.real(x)
        x_imag = sm.imag(x)

        x_real_transpose = x_real.transpose(-1, -2)
        x_imag_transpose = x_imag.transpose(-1, -2)
        x_expected_transpose = sm.stick(x_real_transpose, x_imag_transpose)

        x_result_transpose = sm.transpose(x)

        self.assertAllEqual(x_expected_transpose, x_result_transpose)
        self.assertAllEqual(x, x_result_transpose)   # because they are symmetric matrices

    def test_conj_transpose(self):
        x = get_random_symmetric_matrices(10, 4)
        x_real = sm.real(x)
        x_imag = sm.imag(x)

        x_real_transpose = x_real.transpose(-1, -2)
        x_imag_transpose = x_imag.transpose(-1, -2)
        x_expected_conj_transpose = sm.stick(x_real_transpose, x_imag_transpose * -1)

        x_result_conj_transpose = sm.conj_trans(x)

        self.assertAllEqual(x_expected_conj_transpose, x_result_conj_transpose)

    def test_symmetric_absolute_value2d(self):
        x = torch.ones((3, 2, 2, 2), dtype=torch.float)
        expected = torch.ones((3, 2, 2), dtype=torch.float) * 2
        expected = torch.sqrt(expected)

        self.assertAllClose(expected, sm.sym_abs(x))

    def test_symmetric_absolute_value5d(self):
        x = torch.ones((3, 2, 5, 5), dtype=torch.float)
        expected = torch.ones((3, 5, 5), dtype=torch.float) * 2
        expected = torch.sqrt(expected)

        self.assertAllClose(expected, sm.sym_abs(x))

    def test_add(self):
        x = get_random_symmetric_matrices(10, 4)
        expected = x * 2

        self.assertAllEqual(expected, sm.add(x, x))

    def test_subtract(self):
        x = get_random_symmetric_matrices(10, 4)
        result = torch.zeros_like(x)

        self.assertAllEqual(result, sm.subtract(x, x))

    def test_multiply_by_i(self):
        x = get_random_symmetric_matrices(10, 4)

        result = sm.multiply_by_i(x)
        expected_real = sm.real(result)
        expected_imag = sm.imag(result)

        self.assertAllEqual(expected_real, -sm.imag(x))
        self.assertAllEqual(expected_imag, sm.real(x))

    def test_pow_square(self):
        x_real = torch.Tensor([[[1, -3],
                                [5, -7]]])
        x_imag = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[-80, -112],
                                       [-171, -176]]])
        expected_imag = torch.Tensor([[[18, 66],
                                       [-140, -210]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.pow(x, 2)

        self.assertAllClose(expected, result)

    def test_pow_cube(self):
        x_real = torch.Tensor([[[1, -3],
                                [5, -7]]])
        x_imag = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[-242, 1062],
                                       [-2815, 4382]]])
        expected_imag = torch.Tensor([[[-702, 1034],
                                       [1694, -1170]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.pow(x, 3)

        self.assertAllClose(expected, result)

    def test_pow_inverse(self):
        x_real = torch.Tensor([[[1, -3],
                                [5, -7]]])
        x_imag = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[0.01219512, -0.02307692],
                                       [0.02262443, -0.02554744]]])
        expected_imag = torch.Tensor([[[-0.10975609, 0.08461538],
                                       [0.06334841, -0.05474452]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.pow(x, -1)

        self.assertAllClose(expected, result)

    def test_pow_square_root(self):
        x_real = torch.Tensor([[[1, -3],
                                [5, -7]]])
        x_imag = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[2.24225167, 2.04960413],
                                       [3.15167167, 2.18551428]]])
        expected_imag = torch.Tensor([[[2.00691120, -2.68344501],
                                       [-2.22104353, 3.43168656]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.pow(x, 0.5)

        self.assertAllClose(expected, result)

    def test_bmm(self):
        x_real = torch.Tensor([[[1, -3],
                                [5, -7]]])
        x_imag = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        x = sm.stick(x_real, x_imag)

        y_real = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        y_imag = torch.Tensor([[[1, -3],
                                [5, -7]]])
        y = sm.stick(y_real, y_imag)

        expected_real = torch.Tensor([[[97, -106],
                                       [82, -97]]])
        expected_imag = torch.Tensor([[[221, -246],
                                       [-366, 413]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.bmm(x, y)

        self.assertAllClose(expected, result)

    def test_bmm3(self):
        x_real = torch.Tensor([[[1, -3],
                                [5, -7]]])
        x_imag = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        x = sm.stick(x_real, x_imag)

        y_real = torch.Tensor([[[9, -11],
                                [-14, 15]]])
        y_imag = torch.Tensor([[[1, -3],
                                [5, -7]]])
        y = sm.stick(y_real, y_imag)

        z_real = torch.Tensor([[[-3, -1],
                                [-2, 5]]])
        z_imag = torch.Tensor([[[-1, 3],
                                [0, -2]]])
        z = sm.stick(z_real, z_imag)

        expected_real = torch.Tensor([[[142, -1782],
                                       [-418, 1357]]])
        expected_imag = torch.Tensor([[[-268, -948],
                                       [190, 2871]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.bmm3(x, y, z)

        self.assertAllClose(expected, result)

    def test_inverse_symmetric_2d(self):
        # expected result from https://adrianstoll.com/linear-algebra/matrix-inversion.html
        x_real = torch.Tensor([[[1, -3],
                                [-3, 7]]])
        x_imag = torch.Tensor([[[-9, 11],
                                [11, 15]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[256/8105, 141/16210],
                                       [141/16210, 23/16210]]])
        expected_imag = torch.Tensor([[[921/16210, -356/8105],
                                       [-356/8105, -288/8105]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.inverse(x)

        self.assertAllClose(expected, result)

    def test_inverse_symmetric_3d(self):
        # expected result from https://adrianstoll.com/linear-algebra/matrix-inversion.html
        x_real = torch.Tensor([[[-1, -3,  9],
                                [-3,  5,  7],
                                [ 9,  7, 11]]])
        x_imag = torch.Tensor([[[ 9, 4, -6],
                                [ 4, 7,  9],
                                [-6, 9, -3]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[-36251/845665, -27631/845665, 188/9949],
                                       [-27631/845665, 251611/3382660, -689/39796],
                                       [188/9949, -689/39796, 1299/39796]]])
        expected_imag = torch.Tensor([[[-18757/845665,  -35642/845665, 532/9949],
                                       [-35642/845665, -112703/3382660, -1103/39796],
                                       [532/9949, -1103/39796, 289/39796]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.inverse(x)

        self.assertAllClose(expected, result)

    def test_inverse_nonsymmetric_3d(self):
        # expected result from https://adrianstoll.com/linear-algebra/matrix-inversion.html
        x_real = torch.Tensor([[[-1, -3,  9],
                                [ 3,  5,  7],
                                [ 2,  9, 11]]])
        x_imag = torch.Tensor([[[ 9, 4, -6],
                                [-4, 7,  9],
                                [-2, 7, -3]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[951/16589, 3223/16589, -4496/16589],
                                       [5029/66356, 2486/16589, 1387/33178],
                                       [2137/66356, 1030/16589, -1828/16589]]])
        expected_imag = torch.Tensor([[[-3143/16589, -3622/16589, 1532/16589],
                                       [5533/66356, 6925/33178, -17977/66356],
                                       [-4167/66356, -5779/33178, 6565/66356]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.inverse(x)

        self.assertAllClose(expected, result)

    def test_positive_conjugate_projection_with_positive_eigenvalues(self):
        x = torch.Tensor([[[0.9408, 0.1332],
                           [0.1332, 0.5936]]])

        result = sm.positive_conjugate_projection(x)

        self.assertAllClose(x, result)

    def test_positive_conjugate_projection_with_negative_eigenvalues(self):
        x = torch.Tensor([[[6, 5],
                           [5, 3]]])
        expected = torch.Tensor([[[6.2566, 4.6551],
                                  [4.6551, 3.4636]]])

        result = sm.positive_conjugate_projection(x)

        self.assertAllClose(expected, result, rtol=1e-4)

    def test_to_hermitian(self):
        m = torch.rand(4, 2, 3, 3)

        h = sm.to_hermitian(m)

        h_real, h_imag = sm.real(h), sm.imag(h)

        # 1 - real part is symmetric
        self.assertAllEqual(h_real, h_real.transpose(-1, -2))

        # 2 - Imaginary diagonal must be 0
        imag_diag = torch.diagonal(h_imag, dim1=-2, dim2=-1)
        self.assertAllEqual(imag_diag, torch.zeros_like(imag_diag))

        # 3 - imaginary elements in the upper triangular part of the matrix must be of opposite sign than the
        # elements in the lower triangular part
        imag_triu = torch.triu(h_imag, diagonal=1)
        imag_tril = torch.tril(h_imag, diagonal=-1)
        self.assertAllEqual(imag_triu, imag_tril.transpose(-1, -2) * -1)

    def test_to_compound_real_symmetric_from_hermitian(self):
        x_real = torch.Tensor([[[1, -3],
                                [-3, 7]]])
        x_imag = torch.Tensor([[[0,  5],
                                [-5, 0]]])
        x = sm.stick(x_real, x_imag)

        expected = torch.Tensor([[[ 1, -3,  0, -5],
                                  [-3,  7,  5,  0],
                                  [ 0,  5,  1, -3],
                                  [-5,  0, -3,  7]]])

        result = sm.to_compound_real_symmetric_from_hermitian(x)

        self.assertAllEqual(expected, result)

    def test_to_hermitian_from_compound_real_symmetric_2d(self):
        x = torch.Tensor([[[ 1, -3,  0, -5],
                           [-3,  7,  5,  0],
                           [ 0,  5,  1, -3],
                           [-5,  0, -3,  7]]])

        expected_real = torch.Tensor([[[1, -3],
                                       [-3, 7]]])
        expected_imag = torch.Tensor([[[0, 5],
                                       [-5, 0]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.to_hermitian_from_compound_real_symmetric(x)

        self.assertAllEqual(expected, result)

    def test_to_hermitian_from_compound_real_symmetric_3d(self):
        x = torch.Tensor([[[ 1, -3,  5,  0, -5,  3],
                           [-3,  7, -2, -5,  0,  4],
                           [ 5, -2,  4,  3, -4,  0],
                           [ 0,  5, -3,  1, -3,  5],
                           [-5,  0,  4, -3,  7, -2],
                           [ 3, -4,  0,  5, -2,  4]]])

        expected_real = torch.Tensor([[[ 1, -3,  5],
                                       [-3,  7, -2],
                                       [ 5, -2,  4]]])
        expected_imag = torch.Tensor([[[ 0,  5, -3],
                                       [-5,  0,  4],
                                       [ 3, -4,  0]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.to_hermitian_from_compound_real_symmetric(x)

        self.assertAllEqual(expected, result)

    def test_hermitian_eig(self):
        # expected result from https://www.arndt-bruenner.de/mathe/scripts/engl_eigenwert2.htm
        x_real = torch.Tensor([[[1, -3],
                                [-3, 7]]])
        x_imag = torch.Tensor([[[0, 5],
                                [-5, 0]]])
        x = sm.stick(x_real, x_imag)

        expected = torch.Tensor([[[-2.55743852, 10.55743852]]])

        _, _, result = sm.hermitian_eig(x)

        self.assertAllClose(expected, result)

    def test_hermitian_matrix_sqrt(self):
        # expected result from https://www.wolframalpha.com/
        x_real = torch.Tensor([[[0.9408,  0.1332],
                                [0.1332,  0.5936]]])
        x_imag = torch.Tensor([[[0.0000,  0.5677],
                                [-0.5677,  0.0000]]])
        x = sm.stick(x_real, x_imag)

        expected_real = torch.Tensor([[[0.896153, 0.0847679],
                                       [0.0847679, 0.675196]]])
        expected_imag = torch.Tensor([[[0, 0.361282],
                                       [-0.361282, 0]]])
        expected = sm.stick(expected_real, expected_imag)

        result = sm.hermitian_matrix_sqrt(x)

        self.assertAllClose(expected, result, rtol=1e-05, atol=1e-06)

    def test_matrix_sqrt(self):
        # Expected results: https://www.wolframalpha.com/input/?i=sqrt+%7B%7B0.7047%2C+0.2545%2C+0.0000%2C+-0.1481%7D%2C
        # +%7B0.2545%2C+0.2122%2C+0.1481%2C+0.0000%7D%2C+%7B0.0000%2C+0.1481%2C+0.7047%2C+0.2545%7D%2C+%7B-0.1481%2C
        # +0.0000%2C+0.2545%2C+0.2122%7D%7D
        x = torch.Tensor([[[0.7047, 0.2545, 0.0000, -0.1481],
                           [0.2545, 0.2122, 0.1481, 0.0000],
                           [0.0000, 0.1481, 0.7047, 0.2545],
                           [-0.1481, 0.0000, 0.2545, 0.2122]]])

        expected = torch.Tensor([[[0.802225, 0.213705, 0., -0.12436],
                                  [0.213705, 0.388671, 0.12436, 0],
                                  [0., 0.12436, 0.802225, 0.213705],
                                  [-0.12436, 0, 0.213705, 0.388671]]])

        result = sm.matrix_sqrt(x)

        self.assertAllClose(expected, result, rtol=1e-05, atol=1e-06)


if __name__ == '__main__':
    unittest.main()
