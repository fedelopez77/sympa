import torch
import unittest
from sympa.manifolds import UpperHalfManifold
import sympa.math.symmetric_math as sm
from sympa.utils import set_seed
from tests.utils import get_random_symmetric_matrices, assert_equal, assert_almost_equal


class TestUpperHalfManifold(unittest.TestCase):

    def setUp(self):
        set_seed(42)
        self.manifold = UpperHalfManifold()

    def test_overline_r_equals_a_inverse_r_a(self):
        # \overline(R) = A^-1 R A, A = (ẑ1 - z2)(ẑ1 - ẑ2)^{-1}
        x = self.manifold.projx(get_random_symmetric_matrices(3, 3))
        y = self.manifold.projx(get_random_symmetric_matrices(3, 3))

        conj_x = sm.conjugate(x)
        conj_y = sm.conjugate(y)
        term_a = sm.subtract(conj_x, y)
        term_b = sm.inverse(sm.subtract(conj_x, conj_y))
        a = sm.bmm(term_a, term_b)
        r = self.manifold.r_metric(x, y)

        a_inverse = sm.inverse(a)
        a_inverse_r_a = sm.bmm(a_inverse, r)
        a_inverse_r_a = sm.bmm(a_inverse_r_a, a)
        conj_r = sm.conjugate(r)

        self.assertTrue(assert_almost_equal(conj_r, a_inverse_r_a, rtol=1e-05, atol=1e-05))


if __name__ == '__main__':
    unittest.main()
