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

    def test_caley_transform_real_pos_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)
        x = UpperHalfManifold().projx(x)

        tran_x = caley_transform(x)
        result = inverse_caley_transform(tran_x)

        self.assertAllClose(x, result)

    def test_caley_transform_real_pos_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x), sm.imag(x) * -1)
        x = UpperHalfManifold().projx(x)

        tran_x = caley_transform(x)
        result = inverse_caley_transform(tran_x)

        self.assertAllClose(x, result)

    def test_caley_transform_real_neg_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x) * -1)
        x = UpperHalfManifold().projx(x)

        tran_x = caley_transform(x)
        result = inverse_caley_transform(tran_x)

        self.assertAllClose(x, result)

    def test_caley_transform_real_neg_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))
        x = UpperHalfManifold().projx(x)

        tran_x = caley_transform(x)
        result = inverse_caley_transform(tran_x)

        self.assertAllClose(x, result)

    def test_inverse_caley_transform_real_pos_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)
        x = BoundedDomainManifold().projx(x)

        tran_x = inverse_caley_transform(x)
        result = caley_transform(tran_x)

        self.assertAllClose(x, result)

    def test_inverse_caley_transform_real_pos_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x), sm.imag(x) * -1)
        x = BoundedDomainManifold().projx(x)

        tran_x = inverse_caley_transform(x)
        result = caley_transform(tran_x)

        self.assertAllClose(x, result)

    def test_inverse_caley_transform_real_neg_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x) * -1)
        x = BoundedDomainManifold().projx(x)

        tran_x = inverse_caley_transform(x)
        result = caley_transform(tran_x)

        self.assertAllClose(x, result)

    def test_inverse_caley_transform_real_neg_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))
        x = BoundedDomainManifold().projx(x)

        tran_x = inverse_caley_transform(x)
        result = caley_transform(tran_x)

        self.assertAllClose(x, result)


if __name__ == '__main__':
    unittest.main()
