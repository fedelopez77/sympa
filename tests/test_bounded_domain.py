import torch
import unittest
from sympa.manifolds import BoundedDomainManifold
import sympa.math.compsym_math as sm
from sympa.math.cayley_transform import inverse_cayley_transform
import sympa.tests
from tests.utils import get_random_symmetric_matrices


class TestBoundedDomainManifold(sympa.tests.TestCase):

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float64)
        self.manifold = BoundedDomainManifold(dim=3)

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

    # def test_proj_x_of_point_already_in_the_space_doesnot_affect(self):
    #     x = self.manifold.random(10)
    #
    #     proj_x = self.manifold.projx(x)
    #
    #     # asserts exact equality
    #     self.assertTrue(torch.all(x == proj_x))
    #
    #     # assert symmetry
    #     self.assertAllClose(proj_x, sm.transpose(proj_x))
    #
    #     # assert all points belong to the manifold
    #     for point in proj_x:
    #         self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_random_generator(self):
        x = self.manifold.random(100)

        for point in x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_random_generator_larger_values(self):
        x = self.manifold.random(100) * 10

        for point in x:
            self.assertTrue(self.manifold.check_point_on_manifold(point))

    def test_distance_is_symmetric(self):
        x = self.manifold.random(10)
        y = self.manifold.random(10)

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

    def test_distance_to_same_point_is_zero(self):
        x = self.manifold.random(10)

        dist_xx = self.manifold.dist(x, x)

        self.assertAllClose(dist_xx, torch.zeros_like(dist_xx))


if __name__ == '__main__':
    unittest.main()
