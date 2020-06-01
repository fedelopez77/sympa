
import torch
import sympa.math.symmetric_math as sm


def takagi_factorization(a: torch.Tensor):
    """
    Given A ('a') is a square, complex, symmetric matrix.
    Calculates factorization A = SDS^T
     - D is a real nonnegative diagonal matrix
     - S is unitary

    See https://en.wikipedia.org/wiki/Matrix_decomposition#Takagi's_factorization

    Returns 'eigenvalues' (elements of D) and V as 'eigenvectors'

    :param a: b x 2 x n x n. PRE: Each matrix must be symmetric
    :return: eigenvalues: b x n, eigenvectors: b x 2 x n x n
    """
    # z1, D
    z1, eigenvalues, diagonal = _get_z1(a)            # z1: b x 2 x n x n, evalues: b x n, diagonal: b x 2 x n x n

    z2, b = _get_z2(a, z1)                  # z2, b: b x 2 x n x n. Im(z2) == 0

    # assert that d_i = |b_i|
    assert torch.allclose(torch.diag_embed(eigenvalues), sm.sym_abs(b))

    z3 = _get_z3(b)

    s = sm.bmm3(z1, z2, z3)             # S = Z1 Z2 Z3

    assert s_transpose_a_s_equals_diag(s, a, diagonal)

    return eigenvalues, s


def _get_z1(a: torch.Tensor):
    """
    A^* -> A conjugate transpose
    Z1 (and D): A^* A = Z1^* D Z1
    1 - Build A^* A
    2 - Build [A^* A]: real symmetric
    3 - Diagonalize [A^* A] = [Z1] [D] [Z1]^T
    4 - From [Z1], obtain Z1
         From [D], obtain D
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
    eigvalues_idx = torch.arange(start=0, end=eigenvalues.size(-1), step=2, dtype=torch.long, device=eigenvalues.device)
    z_eigenvalues = eigenvalues.index_select(index=eigvalues_idx, dim=-1)
    # builds diagonal matrix with eigenvalues, without repetition
    diagonal = torch.diag_embed(z_eigenvalues)
    diagonal = sm.stick(diagonal, torch.zeros_like(diagonal))

    # reorders eigenvectors due to repetitions
    eigenvectors = reorder_eigenvectors(eigenvectors, desc_indices)
    z1 = sm.to_hermitian_from_compound_real_symmetric(eigenvectors)

    # asserts: A^* A = Z1^* D Z1
    assert torch.allclose(a_star_a, sm.bmm3(sm.conj_trans(z1), diagonal, z1), rtol=1e-05, atol=1e-06)

    return z1, z_eigenvalues, diagonal


def reorder_eigenvectors(eigenvectors, desc_indices):
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
            if torch.allclose(elem_a, elem_b):
                left.append(vecs[i])
                right.append(vecs[i + 1])
            elif torch.allclose(elem_a, elem_b * -1):   # in this case, it swaps the eigenvectors
                left.append(vecs[i + 1])
                right.append(vecs[i])
            else:
                raise ValueError("Elements in eigenvectors are not equal. Was the original matrix symmetric?")

        left.extend(right)
        reordered = torch.cat(left, dim=-1)             # 2n x 2n
        result.append(reordered)

    result = torch.stack(result, dim=0)
    result = result.transpose(-1, -2)
    return result


def _get_z2(a: torch.Tensor, z1: torch.Tensor):
    """
    1 - Build W = Z1^T A Z1     (Z1 from step 1)
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
    w = sm.bmm3(sm.transpose(z1), a, z1)                                    # b x 2 x n x n

    real_w = sm.real(w)                                                     # b x n x n
    assert torch.allclose(real_w, real_w.transpose(-1, -2))                 # assert Re(W) is symmetric

    # diagonalize Re(W)
    real_b, real_z2 = torch.symeig(real_w, eigenvectors=True)               # real_b: b x n, z2: b x n x n
    # sets values and vectors in descending order
    desc_index = torch.arange(start=real_b.size(-1) - 1, end=-1, step=-1, dtype=torch.long, device=real_b.device)
    real_b = real_b.index_select(dim=-1, index=desc_index)
    real_z2 = real_z2.index_select(dim=-1, index=desc_index)

    # assert that real_z2 is orthogonal: it checks only on the first and last pairs of tensors for simplicity
    assert torch.allclose(torch.sum(real_z2[:, :, 0] * real_z2[:, :, 1]), torch.Tensor(0))
    assert torch.allclose(torch.sum(real_z2[:, :, -2] * real_z2[:, :, -1]), torch.Tensor(0))

    # build Im(B)
    imag_b = torch.bmm(real_z2.transpose(-1, -2), sm.imag(w))
    imag_b = torch.bmm(imag_b, real_z2)                                     # b x n x n



    # TODO: assert that imag_b is diagonal: torch.allclose(imag_b.norm(), torch.diagonal(imag_b, dim1=1, dim2=2).norm())
    # THIS IS NOT HAPPENING




    # stick real and imaginary parts
    z2 = sm.stick(real_z2, torch.zeros_like(real_z2))
    b = sm.stick(torch.diag_embed(real_b), imag_b)

    return z2, b


def _get_z3(b: torch.Tensor):
    """
    Calculates Z3 (diagonal matrix) such that Z3 B Z3 is real, with the same B from Z2
    B is complex diagonal with entries b1, ..., bn.
    Then zi = (sqrt(bi / |bi|))^-1      CAREFUL that each entry bi is a complex number!!!

    :param b: b x 2 x n x n
    :return: z3: b x 2 x n x n
    """
    mod_b = sm.sym_abs(b)                                # b x n x n

    # removes zeros because it will be used for division
    mod_b = torch.where(mod_b != 0, mod_b, torch.ones_like(mod_b))
    compound_mod_b = sm.stick(mod_b, mod_b)

    z3 = b / compound_mod_b                                 # b x 2 x n x n
    z3 = sm.pow(z3, 0.5)
    z3 = sm.pow(z3, -1)

    return z3


def s_transpose_a_s_equals_diag(s, a, diagonal):
    """
    :param s, a, diagonal: b x 2 x n x n
    :return:
    """
    s_transpose_a = sm.bmm(sm.transpose(s), a)
    s_transpose_a_s = sm.bmm(s_transpose_a, s)           # b x 2 x n x n

    return torch.allclose(s_transpose_a_s, diagonal)