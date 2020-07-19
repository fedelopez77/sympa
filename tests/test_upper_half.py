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
        self.manifold = UpperHalfManifold(ndim=3)

    def test_proj_x_real_pos_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_proj_x_real_pos_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x), sm.imag(x) * -1)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_proj_x_real_neg_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_proj_x_real_neg_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x) * -1)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_proj_x_of_point_already_in_the_space_doesnot_affect(self):
        x = self.manifold.random(10)

        proj_x = self.manifold.projx(x)

        # asserts exact equality
        self.assertTrue(torch.all(x == proj_x))

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_random_generator(self):
        x = self.manifold.random(100)

        for point in x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_random_generator_larger_values(self):
        x = self.manifold.random(100) * 10

        for point in x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_distance_is_symmetric_real_pos_imag_pos(self):
        x = self.manifold.random(10) * 10
        y = self.manifold.random(10) * 10

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)

    def test_distance_is_symmetric_real_neg_imag_pos(self):
        x = self.manifold.random(10)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))
        y = self.manifold.random(10)

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)

    def test_distance_to_same_point_is_zero(self):
        x = self.manifold.random(10) * 10

        dist_xx = self.manifold.dist(x, x)

        self.assertAllClose(dist_xx, torch.zeros_like(dist_xx))

    def test_distance_with_small_perturbation(self):
        x = self.manifold.random(10)
        y = x.clone()
        y[:, 0, 0] = y[:, 0, 0] * 1.001

        # In this case, the real part of Z3 is not symmetric any more,

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)


if __name__ == '__main__':
    unittest.main()
