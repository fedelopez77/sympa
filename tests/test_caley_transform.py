import sympa.tests
from sympa.math import symmetric_math as sm
from sympa.math.caley_transform import caley_transform, inverse_caley_transform
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold
from tests.utils import get_random_symmetric_matrices
import unittest
import torch


class TestCaleyTransform(sympa.tests.TestCase):

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float64)
        self.upper_half_manifold = UpperHalfManifold(ndim=3)
        self.bounded_manifold = BoundedDomainManifold(ndim=3)

    def test_caley_transform_real_pos_imag_pos(self):
        x = self.upper_half_manifold.random(10)

        tran_x = caley_transform(x)
        result = inverse_caley_transform(tran_x)

        self.assertAllClose(x, result)
        # the intermediate result belongs to the Bounded domain manifold
        for point in tran_x:
            self.assertTrue(self.bounded_manifold.check_point_on_manifold(point))
        # the final result belongs to the Upper Half Space manifold
        for point in result:
            self.assertTrue(self.upper_half_manifold.check_point_on_manifold(point))

    def test_caley_transform_real_neg_imag_pos(self):
        x = self.upper_half_manifold.random(10)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))

        tran_x = caley_transform(x)
        result = inverse_caley_transform(tran_x)

        self.assertAllClose(x, result)
        # the intermediate result belongs to the Bounded domain manifold
        for point in tran_x:
            self.assertTrue(self.bounded_manifold.check_point_on_manifold(point))
        # the final result belongs to the Upper Half Space manifold
        for point in result:
            self.assertTrue(self.upper_half_manifold.check_point_on_manifold(point))

    def test_caley_transform_from_projx(self):
        x = get_random_symmetric_matrices(10, 3)
        x = self.upper_half_manifold.projx(x)

        tran_x = caley_transform(x)
        result = inverse_caley_transform(tran_x)

        self.assertAllClose(x, result)
        # the intermediate result belongs to the Bounded domain manifold
        for point in tran_x:
            self.assertTrue(self.bounded_manifold.check_point_on_manifold(point))
        # the final result belongs to the Upper Half Space manifold
        for point in result:
            self.assertTrue(self.upper_half_manifold.check_point_on_manifold(point))

    def test_inverse_caley_transform(self):
        x = self.bounded_manifold.random(10)

        tran_x = inverse_caley_transform(x)
        result = caley_transform(tran_x)

        self.assertAllClose(x, result)
        # the intermediate result belongs to the Upper Half Space manifold
        for point in tran_x:
            self.assertTrue(self.upper_half_manifold.check_point_on_manifold(point))
        # the final result belongs to the Bounded domain manifold
        for point in result:
            self.assertTrue(self.bounded_manifold.check_point_on_manifold(point))

    def test_inverse_caley_transform_from_projx(self):
        x = get_random_symmetric_matrices(10, 3)
        x = self.bounded_manifold.projx(x)

        tran_x = inverse_caley_transform(x)
        result = caley_transform(tran_x)

        self.assertAllClose(x, result)
        # the intermediate result belongs to the Upper Half Space manifold
        for point in tran_x:
            self.assertTrue(self.upper_half_manifold.check_point_on_manifold(point))
        # the final result belongs to the Bounded domain manifold
        for point in result:
            self.assertTrue(self.bounded_manifold.check_point_on_manifold(point))


if __name__ == '__main__':
    unittest.main()
