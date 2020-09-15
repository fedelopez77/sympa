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
        self.ndims = 3
        self.manifold = UpperHalfManifold(ndim=self.ndims)

    def test_proj_x_real_pos_imag_pos(self):
        x = get_random_symmetric_matrices(10, self.ndims)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_proj_x_real_pos_imag_neg(self):
        x = get_random_symmetric_matrices(10, self.ndims)
        x = sm.stick(sm.real(x), sm.imag(x) * -1)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_proj_x_real_neg_imag_pos(self):
        x = get_random_symmetric_matrices(10, self.ndims)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert all points belong to the manifold
        for point in proj_x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_proj_x_real_neg_imag_neg(self):
        x = get_random_symmetric_matrices(10, self.ndims)
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

    def test_random_generator_3d(self):
        x = self.manifold.random(100)

        for point in x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_random_generator_larger_values(self):
        x = self.manifold.random(100) * 10

        for point in x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_random_generator_4d(self):
        manifold = UpperHalfManifold(ndim=4)
        x = manifold.random(100)

        for point in x:
            self.assertTrue(manifold.check_point_on_manifold(point))

    def test_random_generator_50d(self):
        manifold = UpperHalfManifold(ndim=50)
        x = manifold.random(100)

        for point in x:
            self.assertTrue(manifold.check_point_on_manifold(point))

    def test_distance_is_symmetric_real_pos_imag_pos(self):
        x = self.manifold.random(10)
        y = self.manifold.random(10)

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
        x = self.manifold.random(10, top=10)

        dist_xx = self.manifold.dist(x, x)

        self.assertAllClose(dist_xx, torch.zeros_like(dist_xx))

    def test_distance_is_small_with_small_perturbation(self):
        x = self.manifold.random(10)
        y = x.clone()
        y[:, 0] = y[:, 0] * 1.001

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)

    def test_distance_is_symmetric_with_small_values(self):
        x = self.manifold.random(10, epsilon=0.0001, top=0.001)
        y = self.manifold.random(10, epsilon=0.0001, top=0.001)

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)

    def test_distance_is_symmetric_with_very_small_values(self):
        x = self.manifold.random(10, epsilon=0.00001, top=0.0001)
        y = self.manifold.random(10, epsilon=0.00001, top=0.0001)

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)

    def test_distance_is_symmetric_with_diagonal_matrices(self):
        x = self.manifold.random(10)
        y = self.manifold.random(10)
        diagonal_mask = torch.eye(self.ndims).unsqueeze(0).repeat(10, 1, 1).bool()
        diagonal_mask = sm.stick(diagonal_mask, diagonal_mask)
        x = torch.where(diagonal_mask, x, torch.zeros_like(x))
        y = torch.where(diagonal_mask, y, torch.zeros_like(y))

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)

    def test_distance_is_symmetric_only_imaginary_matrices(self):
        x = self.manifold.random(10)
        y = self.manifold.random(10)
        zeros = torch.zeros_like(sm.real(x))
        x = sm.stick(zeros, sm.imag(x))
        y = sm.stick(zeros, sm.imag(y))

        dist_xy = self.manifold.dist(x, y)
        dist_yx = self.manifold.dist(y, x)

        self.assertAllClose(dist_xy, dist_yx)


if __name__ == '__main__':
    unittest.main()
