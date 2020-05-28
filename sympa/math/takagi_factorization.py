
import torch
import sympa.math.symmetric_math as smath


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
    # S = Z1 Z2 Z3

    # z1, D
    z1, eigenvalues = _get_z1(a)            # z1: b x 2 x n x n, eigenvalues: b x n
    diag = torch.diag_embed(eigenvalues)    # b x n x n

    z2, b = _get_z2(a, z1)                  # z2, b: b x 2 x n x n. Im(z2) == 0

    # assert that d_i = |b_i|
    assert torch.allclose(diag, smath.sym_abs(b))

    z3 = _get_z3(b)

    s = smath.bmm(z1, z2)
    s = smath.bmm(s, z3)

    assert s_transpose_a_s_equals_diag(s, a, diag)

    return eigenvalues, s


def _get_z1(a: torch.Tensor):
    """
    A^* -> A conjugate transpose
    Z1 (and D): A^* A = Z1^* D^2 Z1
    1 - Build A^* A
    2 - Build [A^* A]: real symmetric
    3 - Diagonalize [A^* A] = [Z1] [D] [Z1]^T
    4 - From [Z1], obtain Z1
         From [D], obtain D
    Check: Z1 is unitary

    :param a: b x 2 x n x n
    :return z1: b x 2 x n x n, eigenvalues: b x n
    """
    a_star = smath.conj_trans(a)
    a_star_a = smath.bmm(a_star, a)
    a_star_a_2n = smath.to_compound_real_symmetric_from_hermitian(a_star_a)

    eigenvalues, eigenvectors = smath.symeig(a_star_a_2n)

    eigenvalues_index = torch.arange(start=0, end=eigenvalues.size(-1), step=2, dtype=torch.long,
                                     device=eigenvalues.device)
    z_eigenvalues = eigenvalues.index_select(index=eigenvalues_index, dim=-1)


    # TODO: Build Z1 from eigenvectors!!!


    return None, z_eigenvalues


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
    w = smath.bmm(smath.transpose(z1), a)
    w = smath.bmm(w, z1)                                        # b x 2 x n x n

    real_w = smath.real(w)                                      # b x n x n
    assert torch.allclose(real_w, real_w.transpose(-1, -2))     # assert Re(W) is symmetric

    # diagonalize Re(W)
    real_b, real_z2 = torch.symeig(real_w, eigenvectors=True)        # real_b: b x n, z2: b x n x n

    # TODO: assert that real_z2 is orthogonal

    # build Im(B)
    imag_b = torch.bmm(real_z2.transpose(-1, -2), smath.imag(w))
    imag_b = torch.bmm(imag_b, real_z2)                              # b x n x n
    # TODO: assert that imag_b is diagonal

    # stick real and imaginary parts
    z2 = smath.stick(real_z2, torch.zeros_like(real_z2))

    b = smath.stick(torch.diag_embed(real_b), imag_b)

    return z2, b


def _get_z3(b: torch.Tensor):
    """
    Calculates Z3 (diagonal matrix) such that Z3 B Z3 is real, with the same B from Z2
    B is complex diagonal with entries b1, ..., bn.
    Then zi = (sqrt(bi / |bi|))^-1      CAREFUL that each entry bi is a complex number!!!

    :param b: b x 2 x n x n
    :return: z3: b x 2 x n x n
    """
    mod_b = smath.sym_abs(b)                                # b x n x n

    # removes zeros because it will be used for division
    mod_b = torch.where(mod_b != 0, mod_b, torch.ones_like(mod_b))
    compound_mod_b = smath.stick(mod_b, mod_b)

    z3 = b / compound_mod_b                                 # b x 2 x n x n
    z3 = smath.pow(z3, 0.5)
    z3 = smath.pow(z3, -1)

    return z3


def s_transpose_a_s_equals_diag(s, a, diag):
    """
    :param s, a: b x 2 x n x n
    :param diag: b x n x n
    :return:
    """
    s_transpose_a = smath.bmm(smath.transpose(s), a)
    s_transpose_a_s = smath.bmm(s_transpose_a, s)           # b x 2 x n x n

    full_diag = smath.stick(diag, torch.zeros_like(diag))   # b x 2 x n x n

    return torch.allclose(s_transpose_a_s, full_diag)