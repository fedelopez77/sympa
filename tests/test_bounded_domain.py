import torch
import unittest
from sympa.manifolds import BoundedDomainManifold
from sympa.manifolds.bounded_domain import get_id_minus_conjugate_z_times_z
import sympa.math.symmetric_math as sm
from sympa.math.takagi_factorization import takagi_factorization
import sympa.tests
from tests.utils import get_random_symmetric_matrices


class TestBoundedDomainManifold(sympa.tests.TestCase):

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float64)
        self.manifold = BoundedDomainManifold()

    def test_proj_x_real_pos_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert Id - ẐZ is symmetric
        id_minus_zz = get_id_minus_conjugate_z_times_z(proj_x)
        self.assertAllClose(id_minus_zz, sm.transpose(id_minus_zz))

        # assert Id - ẐZ is positive definite
        eigenvalues, _ = takagi_factorization(id_minus_zz)
        self.assertTrue(torch.all(eigenvalues > 0.))

    def test_proj_x_real_pos_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x), sm.imag(x) * -1)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert Id - ẐZ is symmetric
        id_minus_zz = get_id_minus_conjugate_z_times_z(proj_x)
        self.assertAllClose(id_minus_zz, sm.transpose(id_minus_zz))

        # assert Id - ẐZ is positive definite
        eigenvalues, _ = takagi_factorization(id_minus_zz)
        self.assertTrue(torch.all(eigenvalues > 0.))

    def test_proj_x_real_neg_imag_pos(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x))

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert Id - ẐZ is symmetric
        id_minus_zz = get_id_minus_conjugate_z_times_z(proj_x)
        self.assertAllClose(id_minus_zz, sm.transpose(id_minus_zz))

        # assert Id - ẐZ is positive definite
        eigenvalues, _ = takagi_factorization(id_minus_zz)
        self.assertTrue(torch.all(eigenvalues > 0.))

    def test_proj_x_real_neg_imag_neg(self):
        x = get_random_symmetric_matrices(10, 3)
        x = sm.stick(sm.real(x) * -1, sm.imag(x) * -1)

        proj_x = self.manifold.projx(x)

        # assert symmetry
        self.assertAllClose(proj_x, sm.transpose(proj_x))

        # assert Id - ẐZ is symmetric
        id_minus_zz = get_id_minus_conjugate_z_times_z(proj_x)
        self.assertAllClose(id_minus_zz, sm.transpose(id_minus_zz))

        # assert Id - ẐZ is positive definite
        eigenvalues, _ = takagi_factorization(id_minus_zz)
        self.assertTrue(torch.all(eigenvalues > 0.))


if __name__ == '__main__':
    unittest.main()
