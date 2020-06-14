import torch
import unittest
import sympa.math.symmetric_math as sm
from sympa.math.takagi_factorization import takagi_factorization
import sympa.tests
from tests.utils import get_random_symmetric_matrices


class TestTakagiFactorization(sympa.tests.TestCase):

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float64)

    def test_takagi_factorization_real_pos_imag_pos(self):
        a = get_random_symmetric_matrices(3, 3)

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.transpose(s)))

    def test_takagi_factorization_real_pos_imag_pos_inverse(self):
        a = get_random_symmetric_matrices(3, 3)

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.inverse(s)))

    def test_takagi_factorization_real_pos_imag_neg(self):
        a = get_random_symmetric_matrices(3, 3)
        a = sm.stick(sm.real(a), sm.imag(a) * -1)

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.transpose(s)))

    def test_takagi_factorization_real_neg_imag_pos(self):
        a = get_random_symmetric_matrices(3, 3)
        a = sm.stick(sm.real(a) * -1, sm.imag(a))

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.transpose(s)))

    def test_takagi_factorization_real_neg_imag_neg(self):
        a = get_random_symmetric_matrices(3, 3)
        a = sm.stick(sm.real(a) * -1, sm.imag(a) * -1)

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.transpose(s)))

    def test_takagi_factorization_small_values(self):
        a = get_random_symmetric_matrices(3, 3) / 10

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.transpose(s)))

    def test_takagi_factorization_large_values(self):
        a = get_random_symmetric_matrices(3, 3) * 10

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.transpose(s)))

    def test_takagi_factorization_very_large_values(self):
        a = get_random_symmetric_matrices(3, 3) * 1000

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.transpose(s)))

    def test_takagi_factorization_very_large_values_inverse(self):
        a = get_random_symmetric_matrices(3, 3) * 1000

        eigenvalues, s = takagi_factorization(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(s, diagonal, sm.inverse(s)))


if __name__ == '__main__':
    unittest.main()
