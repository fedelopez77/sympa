import torch
import unittest
import sympa.math.symmetric_math as sm
from sympa.math.takagi_factorization import TakagiFactorization
import sympa.tests
from tests.utils import get_random_symmetric_matrices


class TestTakagiFactorization(sympa.tests.TestCase):

    def setUp(self):
        super().setUp()
        torch.set_default_dtype(torch.float64)

    def test_takagi_factorization_real_pos_imag_pos(self):
        a = get_random_symmetric_matrices(3, 3)

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_real_pos_imag_neg(self):
        a = get_random_symmetric_matrices(3, 3)
        a = sm.stick(sm.real(a), sm.imag(a) * -1)

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_real_neg_imag_pos(self):
        a = get_random_symmetric_matrices(3, 3)
        a = sm.stick(sm.real(a) * -1, sm.imag(a))

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_real_neg_imag_neg(self):
        a = get_random_symmetric_matrices(3, 3)
        a = sm.stick(sm.real(a) * -1, sm.imag(a) * -1)

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_small_values(self):
        a = get_random_symmetric_matrices(3, 3) / 10

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_large_values(self):
        a = get_random_symmetric_matrices(3, 3) * 10

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_very_large_values(self):
        a = get_random_symmetric_matrices(3, 3) * 1000

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_real_identity(self):
        a = sm.identity_like(get_random_symmetric_matrices(3, 3))

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))
        self.assertAllClose(a, s)
        self.assertAllClose(torch.ones_like(eigenvalues), eigenvalues)

    def test_takagi_factorization_imag_identity(self):
        a = sm.identity_like(get_random_symmetric_matrices(3, 3))
        a = sm.stick(sm.imag(a), sm.real(a))

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))

    def test_takagi_factorization_real_diagonal(self):
        a = get_random_symmetric_matrices(3, 3) * 10
        a = torch.where(sm.identity_like(a) == 1, a, torch.zeros_like(a))

        eigenvalues, s = TakagiFactorization(3).factorize(a)

        diagonal = torch.diag_embed(eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        self.assertAllClose(a, sm.bmm3(sm.conjugate(s), diagonal, sm.conj_trans(s)))
        # real part of eigenvectors is made of vectors with one 1 and all zeros
        real_part = torch.sum(torch.abs(sm.real(s)), dim=-1)
        self.assertAllClose(torch.ones_like(real_part), real_part)
        # imaginary part of eigenvectors is all zeros
        self.assertAllClose(torch.zeros(1), torch.sum(sm.imag(s)))


if __name__ == '__main__':
    unittest.main()
