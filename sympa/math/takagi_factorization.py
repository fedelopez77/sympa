
import torch
from sympa.config import DEVICE
import sympa.math.symmetric_math as sm
from sympa.utils import row_sort, assert_all_close, get_logging

log = get_logging()


class TakagiFactorization:
    def __init__(self, matrix_rank):
        self.rank = matrix_rank
        self.eigvalues_idx = torch.arange(start=0, end=2 * matrix_rank, step=2, dtype=torch.long, device=DEVICE)
        self.zero = torch.Tensor(0).to(DEVICE)

    def factorize(self, a: torch.Tensor):
        """
        Given A ('a') square, complex, symmetric matrix.
        Calculates factorization A = Ŝ D S^*
         - D is a real nonnegative diagonal matrix
         - S is unitary
         - Ŝ: S conjugate
         - S^*: S conjugate transpose

        See https://en.wikipedia.org/wiki/Matrix_decomposition#Takagi's_factorization

        Returns 'eigenvalues' (elements of D) and V as 'eigenvectors'

        :param a: b x 2 x n x n. PRE: Each matrix must be symmetric
        :return: eigenvalues: b x n, eigenvectors: b x 2 x n x n
        """
        # z1, D
        z1, eigenvalues, diagonal = self._get_z1(a)          # z1: b x 2 x n x n, evalues: b x n, diagonal: b x 2 x n x n
        diagonal = sm.pow(diagonal, 0.5)
        eigenvalues = eigenvalues**0.5

        z2, b = self._get_z2(a, z1)                          # z2, b: b x 2 x n x n

        # assert that d_i = |b_i|
        # assert assert_all_close(sm.sym_abs(diagonal), sm.sym_abs(b))

        z3 = self._get_z3(b)

        s = sm.bmm3(sm.conj_trans(z1), z2, z3)             # S = Z1^* Z2 Z3

        # assert self.s_transpose_a_s_equals_diag(s, a, diagonal)

        return eigenvalues, s

    def _get_z1(self, a: torch.Tensor):
        """
        A^* -> A conjugate transpose
        Z1 (and D): A^* A = Z1^* D Z1
        1 - Build A^* A
        2 - Build [A^* A]: real symmetric
        3 - Diagonalize [A^* A] = [S] [D] [S]^T
        4 - From [S], obtain Z1: [S]^T = [Z1]
             From [Deigenvalues.device], obtain D
        Check: Z1 is unitary

        :param a: b x 2 x n x n
        :return z1: b x 2 x n x n, eigenvalues: b x n, diagonal: b x 2 x n x n
        """
        a_star = sm.conj_trans(a)
        a_star_a = sm.bmm(a_star, a)
        a_star_a_2n = sm.to_compound_real_symmetric_from_hermitian(a_star_a)        # b x 2n x 2n

        eigenvalues, eigenvectors = sm.symeig(a_star_a_2n)              # b x 2n; b x 2n x 2n

        eigenvalues, desc_indices = torch.sort(eigenvalues, dim=-1, descending=True)

        # chooses one eigenvalue for every pair since they are repeated
        z_eigenvalues = eigenvalues.index_select(index=self.eigvalues_idx, dim=-1)
        # builds diagonal matrix with eigenvalues, without repetition
        diagonal = torch.diag_embed(z_eigenvalues)
        diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

        # if all eigenvalues equal 1, the input was the identity and we do not need to reorder the eigenvalues
        if torch.any(z_eigenvalues != 1):
            # reorders eigenvectors due to repetitions
            eigenvectors = self.reorder_eigenvectors(eigenvectors, desc_indices)
            eigenvectors = eigenvectors.transpose(-1, -2)
        z1 = sm.to_hermitian_from_compound_real_symmetric(eigenvectors)

        # asserts: A^* A = Z1^* D Z1
        # assert assert_all_close(a_star_a, sm.bmm3(sm.conj_trans(z1), diagonal, z1), factor=2)

        return z1, z_eigenvalues, diagonal

    def reorder_eigenvectors(self, eigenvectors, desc_indices):
        """
        Eigenvectors are repeated according to eigenvalues, so they must be reordered.
        The final matrix should have a block shape of:
        [Z1] =  [[(Re(Z1), -Im(Z1)],
                 [(Im(Z1),  Re(Z1)]]
        :param eigenvectors: b x 2n x 2n
        :return:
        """
        result = []
        half_elem = int(eigenvectors.size(-1) / 2)
        for mat_idx, matrix in enumerate(eigenvectors):     # 2n x 2n. Each matrix can have a different rearrangement
            matrix = matrix.index_select(dim=1, index=desc_indices[mat_idx])    # reorders to descending, as eigvalues
            vecs = matrix.split(1, dim=-1)                  # list of 2n eigenvectors with vecs of 2n x 1
            left, right = [], []
            for i in range(0, len(vecs), 2):
                elem_a = vecs[i][0]
                elem_b = vecs[i + 1][half_elem]
                if assert_all_close(elem_a, elem_b, factor=5) and ((elem_a > 0 and elem_b > 0) or (elem_a < 0 and elem_b < 0)):
                    left.append(vecs[i])
                    right.append(vecs[i + 1])
                elif assert_all_close(elem_a, elem_b * -1, factor=5):  # in this case, it swaps the eigenvectors
                    left.append(vecs[i + 1])
                    right.append(vecs[i])
                else:
                    raise ValueError(f"Elements in eigenvectors are not equal. Was the original matrix symmetric?"
                                     f"Matrix: {matrix}")

            left.extend(right)
            reordered = torch.cat(left, dim=-1)             # 2n x 2n
            result.append(reordered)

        result = torch.stack(result, dim=0)
        return result

    def _get_z2(self, a: torch.Tensor, z1: torch.Tensor):
        """
        1 - Build W = Ẑ1 A Z1^*     (Z1 from step 1)
        The real part of W should be symmetric
        2 - Diagonalize Real part of W: Re(W) = Z2 Re(B) Z2^T
        Check: Z2 is orthogonal
        3 - Build Im(B) = Z2^T Im(W) Z2
        4 - Build B = Re(B); Im(B) (B will be used in step 3)

        :param a: b x 2 x n x n
        :param z1: b x 2 x n x n
        :return: z2: b x 2 x n x n, b: b x 2 x n x n
        """
        # build W
        w = sm.bmm3(sm.conjugate(z1), a, sm.conj_trans(z1))                                    # b x 2 x n x n

        real_w = sm.real(w)                                                     # b x n x n
        assert assert_all_close(real_w, real_w.transpose(-1, -2))  # assert Re(W) is symmetric

        # diagonalize Re(W)
        real_b, real_z2 = torch.symeig(real_w, eigenvectors=True)               # real_b: b x n, z2: b x n x n

        # assert that real_z2 is orthogonal: it checks only on the first and last pairs of tensors for simplicity
        assert torch.allclose(torch.sum(real_z2[:, :, 0] * real_z2[:, :, 1]), self.zero)
        assert torch.allclose(torch.sum(real_z2[:, :, -2] * real_z2[:, :, -1]), self.zero)

        # build Im(B) = Z2^T Im(W) Z2
        imag_b = real_z2.transpose(-1, -2).bmm(sm.imag(w)).bmm(real_z2)     # b x n x n

        # assert that imag_b is diagonal
        assert torch.allclose(imag_b.norm(), torch.diagonal(imag_b, dim1=-2, dim2=-1).norm())

        # reorder |B_i|^2 in descending order
        sorted_b, indexes = self.set_descending_order_bi_sq(torch.diag_embed(real_b), imag_b)   # b x 2 x n x n; b x n

        # reorder z2 according to indices taken from B reordering
        sorted_real_z2 = self.reorder_z2(real_z2, indexes)

        # stick real and imaginary parts
        z2 = sm.stick(sorted_real_z2, torch.zeros_like(real_z2))

        return z2, sorted_b

    def reorder_z2(self, real_z2, indexes):
        """
        Indices contains in each row the index to reorder the columns of real_z2 (batch-wise)
        :param real_z2: b x n x n
        :param indexes: b x n
        :return: real_z2 with columns sorted according to indexes
        """
        result = [i_real_z2.index_select(dim=-1, index=i_index) for i_real_z2, i_index in zip(real_z2, indexes)]
        sorted_real_z2 = torch.stack(result, dim=0)
        return sorted_real_z2

    def set_descending_order_bi_sq(self, real_b, imag_b):
        """
        B = sm.stick(real_b, imag_b). Reorders B such that |B_i|^2 is in descending order.
        Both real and imag parts are diagonal because B is diagonal
        :param real_b, imag_b: b x n x n: Pre: they are diagonal matrixes
        :return: sorted B and indices of reordering
        """
        b = sm.stick(real_b, imag_b)
        sq_mod_b = torch.pow(sm.sym_abs(b), 2)                                          # b x n x n
        diag_sq_mod_b = torch.diagonal(sq_mod_b, offset=0, dim1=-2, dim2=-1)            # b x n
        _, indexes = torch.sort(diag_sq_mod_b, dim=-1, descending=True)

        diag_real_b = torch.diagonal(real_b, offset=0, dim1=-2, dim2=-1)
        diag_imag_b = torch.diagonal(imag_b, offset=0, dim1=-2, dim2=-1)
        sorted_diag_real_b = row_sort(diag_real_b, indexes)
        sorted_diag_imag_b = row_sort(diag_imag_b, indexes)
        sorted_b = sm.stick(torch.diag_embed(sorted_diag_real_b), torch.diag_embed(sorted_diag_imag_b))
        return sorted_b, indexes

    def _get_z3(self, b: torch.Tensor):
        """
        Calculates Z3 (diagonal matrix) such that Z3 B Z3 is real, with the same B from Z2
        B is complex diagonal with entries b1, ..., bn.
        Then zi = (sqrt(bi / |bi|))^-1      CAREFUL that each entry bi is a complex number!!!

        :param b: b x 2 x n x n
        :return: z3: b x 2 x n x n
        """
        # sets values outside of the diagonal to 1 for numerical stability of operations
        diagonal_mask = sm.real(sm.identity_like(b))
        diagonal_mask = sm.stick(diagonal_mask, diagonal_mask)
        b_no_zeros = torch.where(diagonal_mask > 0.5, b, torch.ones_like(b))

        mod_b = sm.sym_abs(b_no_zeros)                                # b x n x n
        compound_mod_b = sm.stick(mod_b, mod_b)

        z3 = b_no_zeros / compound_mod_b
        z3 = sm.pow(z3, 0.5)
        z3 = sm.pow(z3, -1)

        # builds final z3 using only values from the diagonal
        z3 = torch.diag_embed(torch.diagonal(z3, offset=0, dim1=-2, dim2=-1))       # b x 2 x n x n

        # asserts Z3 B Z3 is real
        z3_b_z3 = sm.bmm3(z3, b, z3)
        z3_b_z3_imag = sm.imag(z3_b_z3)
        assert torch.all(z3_b_z3_imag < sm.EPS[z3.dtype])

        return z3

    def s_transpose_a_s_equals_diag(self, s, a, diagonal):
        """
        Asserts S^T A S == D
        :param s, a, diagonal: b x 2 x n x n
        """
        s_transpose_a_s = sm.bmm3(sm.transpose(s), a, s)
        ok = assert_all_close(s_transpose_a_s, diagonal)
        if not ok:
            log.info(f"s_transpose_a_s:{s_transpose_a_s}\n\ndiagonal:{diagonal}\n\n"
                     f"diff: {(s_transpose_a_s - diagonal).abs()}")
        return ok
