import torch
import unittest
from sympa.manifolds import UpperHalfManifold
from sympa.manifolds.upper_half import generate_matrix_in_upper_half_space
import sympa.math.symmetric_math as sm
import sympa.tests
from tests.utils import get_random_symmetric_matrices


class TestUpperHalfManifold(sympa.tests.TestCase):

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float64)
        self.manifold = UpperHalfManifold()

    # def test_overline_r_equals_a_inverse_r_a(self):
    #     # \overline(R) = A^-1 R A, A = (ẑ1 - z2)(ẑ1 - ẑ2)^{-1}
    #     x = self.manifold.projx(get_random_symmetric_matrices(3, 3))
    #     y = self.manifold.projx(get_random_symmetric_matrices(3, 3))
    #
    #     conj_x = sm.conjugate(x)
    #     conj_y = sm.conjugate(y)
    #     term_a = sm.subtract(conj_x, y)
    #     term_b = sm.inverse(sm.subtract(conj_x, conj_y))
    #     a = sm.bmm(term_a, term_b)
    #     r = self.manifold.r_metric(x, y)
    #
    #     a_inverse_r_a = sm.bmm3(sm.inverse(a), r, a)
    #     conj_r = sm.conjugate(r)
    #
    #     self.assertAllClose(conj_r, a_inverse_r_a, rtol=1e-05, atol=1e-05)

    def test_proj_x_real_pos_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert imag(x) is positive definite
        imag_proj_x = sm.imag(proj_x)
        eigenvalues, _ = torch.symeig(imag_proj_x)
        self.assertTrue(torch.all(eigenvalues > 0.))

    def test_proj_x_real_pos_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x), sm.imag(x) * -1)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert imag(x) is positive definite
        imag_proj_x = sm.imag(proj_x)
        eigenvalues, _ = torch.symeig(imag_proj_x)
        self.assertTrue(torch.all(eigenvalues > 0.))

    def test_proj_x_real_neg_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert imag(x) is positive definite
        imag_proj_x = sm.imag(proj_x)
        eigenvalues, _ = torch.symeig(imag_proj_x)
        self.assertTrue(torch.all(eigenvalues > 0.))

    def test_proj_x_real_neg_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x) * -1)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert imag(x) is positive definite
        imag_proj_x = sm.imag(proj_x)
        eigenvalues, _ = torch.symeig(imag_proj_x)
        self.assertTrue(torch.all(eigenvalues > 0.))

    def test_distance_is_symmetric_real_pos_imag_pos(self):
        x = generate_matrix_in_upper_half_space(10, 3) * 10
        y = generate_matrix_in_upper_half_space(10, 3) * 10

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)

    def test_distance_is_symmetric_real_neg_imag_pos(self):
        x = generate_matrix_in_upper_half_space(10, 3) * 10
        x = sm.stick(sm.real(x) * -1, sm.imag(x))
        y = generate_matrix_in_upper_half_space(10, 3) * 10

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)


if __name__ == '__main__':
    unittest.main()
