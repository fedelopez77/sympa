"""Code adapted from https://github.com/hajifkd/takagi_fact/blob/master/takagi_fact/__init__.py"""
import torch
import sympa.math.csym_math as sm


class TakagiFactorization:
    """
    Given a complex symmetric matrix A, the Takagi factorization is an algorithm that
    computes a real diagonal matrix D and a complex unitary matrix S such that
        A = Ŝ D S^*
    where:
        Ŝ: S conjugate
        S^*: S conjugate transpose
    """
    def __init__(self, matrix_rank, use_xitorch=False, return_eigenvectors=True):
        """
        :param matrix_rank:
        :param use_xitorch: if True it uses sm.xitorch_symeig to find eigenvectors and
        eigenvalues of a matrix. If False it uses sm.symeig.
        :param return_eigenvectors: whether to return 'evalues, evecs' or just 'evalues'
        """
        self.rank = matrix_rank
        self.symmetric_svd = self.symmetric_svd_with_eigenvectors if return_eigenvectors \
            else self.symmetric_svd_without_eigenvectors
        self.symeig = sm.xitorch_symeig if use_xitorch else sm.symeig

    def factorize(self, a: torch.Tensor):
        """
        Given A ('a') square, complex, symmetric matrix.
        Calculates factorization such that A = Ŝ D S^*
         - D is a real nonnegative diagonal matrix
         - S is unitary
         - Ŝ: S conjugate
         - S^*: S conjugate transpose

        See https://en.wikipedia.org/wiki/Matrix_decomposition#Takagi's_factorization

        Returns 'eigenvalues' (elements of D) and S as 'eigenvectors'

        :param a: b x 2 x n x n. PRE: Each matrix must be symmetric
        :return: eigenvalues: b x n, eigenvectors: b x 2 x n x n
        """
        return self.symmetric_svd(a)

    def symmetric_svd_with_eigenvectors(self, z: torch.Tensor):
        """
        :param z: complex symmetric matrix
        :return:
        """
        compound_z = sm.to_compound_symmetric(z)    # b x 2 x 2n x 2n. Z = A + iB, then [(A, B),(B, -A)]

        evalues, q = self.symeig(compound_z, eigenvectors=True)    # b x n in ascending order, b x 2n x 2n

        # I can think of Q as 4 n x n matrices.
        # Q = [(X,  Re(U)),
        #      (Y, -Im(U))]     where X, Y are irrelevant and I need to build U
        real_u_on_top_of_minus_imag_u = torch.chunk(q, 2, dim=-1)[-1]
        real_u, minus_imag_u = torch.chunk(real_u_on_top_of_minus_imag_u, 2, dim=-2)
        u = sm.stick(real_u, -minus_imag_u)     # b x 2 x n x n

        sing_values = evalues[:, z.shape[-1]:]
        # sing_values_matrix = sm.bmm3(sm.transpose(u), z, u)     # b x 2 x n x n. imag part should be zero
        # sing_values = torch.diagonal(sm.real(sing_values_matrix), offset=0, dim1=-2, dim2=-1)
        return sing_values, u

    def symmetric_svd_without_eigenvectors(self, z: torch.Tensor):
        """
        :param z: b x 2 x n x n complex symmetric matrix
        :return: eigenvalues: b x n
        """
        compound_z = sm.to_compound_symmetric(z)    # b x 2 x 2n x 2n. Z = A + iB, then [(A, B),(B, -A)]
        # If I want to compute backward over this function I have to compute the eigenvectors anyway
        evalues, _ = self.symeig(compound_z, eigenvectors=True)    # b x n in ascending order
        sing_values = evalues[:, z.shape[-1]:]
        return sing_values
