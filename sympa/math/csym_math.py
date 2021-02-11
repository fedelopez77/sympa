# Definition of operations for points represented as symmetric matrices with complex entries.
# All operations (unless specified otherwise) assume that tensors are of the shape (b, *, 2, n, n), which represent
# a matrix with complex entries.
# b: batch size
# *: other dimensions
# 2: element in position 0 represents real part of the complex entry
#    element in position 1 represents imaginary part of the complex entry
# n: dimensions of the matrix

import torch
from sympa.config import EPS
from geoopt.linalg.batch_linalg import sym as squared_to_symmetric


def real(z: torch.Tensor):
    """
    Returns the real part of z
    :param z: b x 2 x n x n
    :return real: b x * x n x n
    """
    return z[:, 0]


def imag(z: torch.Tensor):
    """
    Returns the imaginary part of z
    :param z: b x 2 x n x n
    :return real: b x * x n x n
    """
    return z[:, 1]


def stick(real_part: torch.Tensor, imag_part: torch.Tensor):
    """
    Returns a symmetric complex tensor made of the real and imaginary parts

    :param real_part: b x n x n
    :param imag_part: b x n x n
    :return z: b x 2 x n x n
    """
    return torch.stack((real_part, imag_part), dim=1)


def conjugate(z: torch.Tensor):
    """Complex conjugate of z"""
    return stick(real(z), -imag(z))


def transpose(z: torch.Tensor):
    return z.transpose(-1, -2)


def conj_trans(z: torch.Tensor):
    """Conjugate transpose of z"""
    return transpose(conjugate(z))


def sym_abs(z: torch.Tensor):
    """Absolute value of z"""
    return torch.sqrt(real(z) ** 2 + imag(z) ** 2)


def add(x: torch.Tensor, y: torch.Tensor):
    return x + y


def subtract(x: torch.Tensor, y: torch.Tensor):
    return x - y


def pow(z: torch.Tensor, exponent):
    """
    z = a + ib = r (cosθ + i sinθ) where r^2 = a^2 + b^2 and tanθ = b / a.
    Then: (a + ib)^n = r^n (cos(nθ) + i sin(nθ)).

    :param z: tensor of b x 2 x n x n
    :param exponent: number (int or float)
    :return z^exponent: b x 2 x n x n
    """
    r = sym_abs(z)
    r = torch.pow(r, exponent)

    tita = torch.atan2(imag(z), real(z))
    tita = tita * exponent

    real_part = r * torch.cos(tita)
    imag_part = r * torch.sin(tita)
    return stick(real_part, imag_part)


def bmm(x: torch.Tensor, y: torch.Tensor):
    """
    Performs a batch matrix-matrix product of matrices stored in x and y.
    If x is a b × 2 x n × m and y is a b × 2 x m × p,
    out will be a b × 2 x n × p.

    Complex product:
        x = a + ib; y = c + id
        xy = (a + ib)(c + id)
           = ac + iad + ibc - bd
           = (ac - bd) + i(ad + bc)

    :param x: b x 2 x n x m
    :param y: b x 2 x m x p
    :return b x 2 x n x p
    """
    real_x, imag_x = real(x), imag(x)
    real_y, imag_y = real(y), imag(y)
    ac = real_x.bmm(real_y)
    bd = imag_x.bmm(imag_y)
    ad = real_x.bmm(imag_y)
    bc = imag_x.bmm(real_y)
    out_real = ac - bd
    out_imag = ad + bc
    return stick(out_real, out_imag)


def bmm3(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """
    Combines the product of 3 matrices in only one function call: A = X Y Z

    :param x: b x 2 x n x m
    :param y: b x 2 x m x p
    :param z: b x 2 x p x q
    :return b x 2 x n x q
    """
    xy = bmm(x, y)
    return bmm(xy, z)


def to_symmetric(y: torch.Tensor):
    """
    Copies the values on the upper triangular to the lower triangular in order to make it symmetric
    :param y: b x * x 2 x n x n
    """
    real_sym = squared_to_symmetric(real(y))
    imag_sym = squared_to_symmetric(imag(y))
    return stick(real_sym, imag_sym)


def diag_embed(d: torch.Tensor):
    """
    Set the n values in d, as the real main diagonal of a n x n matrix.
    The imaginary part is set to zero.
    :param d: b x n
    :return: b x 2 x n x n
    """
    d = torch.diag_embed(d)
    return stick(d, torch.zeros_like(d))


def is_complex_symmetric(x: torch.Tensor, atol=1e-05, rtol=1e-5):
    """
    Returns whether the complex symmetric matrices are symmetric or not
    :param x: b x 2 x n x n
    :param atol: parameter for allclose comparison
    :param rtol: parameter for allclose comparison
    """
    real_x, imag_x = real(x), imag(x)
    return torch.allclose(real_x, real_x.transpose(-1, -2), atol=atol, rtol=rtol) and \
           torch.allclose(imag_x, imag_x.transpose(-1, -2), atol=atol, rtol=rtol)


def multiply_by_i(z: torch.Tensor):
    """
    For Z = X + iY, calculates the operation i Z = i (X + iY) = -Y + iX
    :param z: b x * x 2 x n x n
    """
    return stick(-imag(z), real(z))


def trace(x: torch.Tensor, keepdim=False):
    """
    Batched version of trace for 2D matrices
    :param x: tensor of squared matrices: b x n x n
    :param keepdim: returns tensor of shape (b,) or of shape (b, 1)
    :return: sum of the elements of the diagonal: b x 1 (or (b,))
    """
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1, keepdim=keepdim)


def repr(z: torch.Tensor):
    batch_size, _, n, _ = z.size()
    real_x, imag_x = real(z), imag(z)
    result = []
    for b_i in range(batch_size):
        real_item, imag_item = real_x[b_i], imag_x[b_i]
        rows = []
        for i in range(n):
            row = [f"{real_item[i][j]:.4f}{'+' if imag_item[i][j] >= 0 else ''}{imag_item[i][j]:.4f}j" for j in range(n)]
            rows.append("    ".join(row))
        result.append("\n".join(rows))
        result.append("")
    return "\n".join(result)


def inverse(z: torch.Tensor):
    """
    It calculates the inverse of a matrix with complex entries according to the algorithm explained in:
    "Method to Calculate the Inverse of a Complex Matrix using Real Matrix Inversion" by Andreas Falkenberg
    https://pdfs.semanticscholar.org/f278/b548b5121fd0d09c2e589439b97fad16ece3.pdf

    Given a squared matrix with complex entries Z, such that Z = A + iC where A,C are squared matrices with real
    entries, the algorithm returns M = U + iV, where M = Inverse(Z) and U,V are squared matrices with real entries
    Steps:
        R = inverse(A) * C
        U = inverse(C * R + A)
        V = - (R * U)

    * denotes matrix multiplication

    :param z: b x * x 2 x n x n
    """
    a, c = real(z), imag(z)
    if torch.all(a == 0):
        imag_inverse = torch.inverse(c)
        return stick(a, imag_inverse)
    if torch.all(c == 0):
        real_inverse = torch.inverse(a)
        return stick(real_inverse, c)

    r = torch.inverse(a).bmm(c)
    u = torch.inverse(c.bmm(r) + a)
    v = (r.bmm(u)) * -1
    return stick(u, v)


def positive_conjugate_projection(y: torch.Tensor):
    """In some symmetric spaces, we need to ensure that a matrix Y is positive definite.
    For example, in the Upper half space manifold, Y = Im(Z).
    Since the matrix Y is symmetric, it is possible to diagonalize it.
    For a diagonal matrix the condition is just that the diagonal entries are positive, so we clamp the values
    that are <=0 in the diagonal to epsilon, and then restore the matrix back into non-diagonal form using
    the base change matrix that was obtained from the diagonalization.

    Steps to project: Y = Symmetric Matrix
    1) Y = SDS^-1
    2) D_tilde = clamp(D, min=epsilon)
    3) Y_tilde = SD_tildeS^-1

    :param y: b x n x n. PRE: Each matrix must be symmetric

    """
    eigenvalues, s = symeig(y)
    eigenvalues_tilde = torch.clamp(eigenvalues, min=EPS[y.dtype])
    d_tilde = torch.diag_embed(eigenvalues_tilde)
    y_tilde = s.bmm(d_tilde).bmm(s.inverse())

    # we do this so no operation is applied on the matrices that already belong to the space.
    # This prevents modifying values due to numerical instabilities/floating point ops
    batch_wise_mask = torch.all(eigenvalues > EPS[y.dtype], dim=-1, keepdim=True)    # True means it must not be projected
    mask = batch_wise_mask.unsqueeze(-1).expand_as(y)

    return torch.where(mask, y, y_tilde), batch_wise_mask


def symeig(y: torch.Tensor):
    """
    If Y is a squared symmetric matrix, then Y can be decomposed (diagonalized) as Y = SDS^-1
    where S is the matrix composed by the eigenvectors of Y (S = [v_1, v_2, ..., v_n] where v_i are the
    eigenvectors), D is the diagonal matrix constructed from the corresponding eigenvalues, and S^-1 is the
    matrix inverse of S.

    :param y: b x * x n x n. PRE: Each matrix must be symmetric
    :return: eigenvalues (in ascending order): b x * x n
    :return: eigenvectors: b x * x n x n
    """
    eigenvalues, eigenvectors = torch.symeig(y, eigenvectors=True)      # evalues are in ascending order
    return eigenvalues, eigenvectors


def xitorch_symeig(y: torch.Tensor):
    """
    Idem symeig but using xitorch.
    This is done in this way since according to pytorch documentation for symeig
    (https://pytorch.org/docs/stable/generated/torch.symeig.html):
        Extra care needs to be taken when backward through outputs.
        Such operation is really only stable when all eigenvalues are distinct.
        Otherwise, NaN can appear as the gradients are not properly defined.

    Therefore, the xitorch implementation has a workaround to deal with degenerate
    matrices (matrices with repeated eigenvalues) in the backward pass.

    :param y: b x * x n x n. PRE: Each matrix must be symmetric
    :return: eigenvalues (in ascending order): b x * x n
    :return: eigenvectors: b x * x n x n
    """
    import xitorch
    from xitorch.linalg import symeig as xit_symeig_op
    linop = xitorch.LinearOperator.m(y)
    eigenvalues, eigenvectors = xit_symeig_op(linop,
                                              bck_optioins={"degen_atol": 1e-22, "degen_rtol": 1e-22},
                                              neig=y.shape[-1], method="custom_exacteig", max_niter=1000)
    return eigenvalues, eigenvectors


def identity(dims: int):
    """
    Returns the Identity matrix I, in the shape of 2 x n x n where I[0] is the identity of n x n with real entries
    and I[1] is a matrix of zeros.
    :param dims:
    :return: I: 2 x n x n
    """
    real = torch.eye(dims)
    imag = torch.zeros_like(real)
    return torch.stack((real, imag))


def identity_like(z: torch.Tensor):
    """
    Return an identity of the shape of z, with the same type and device.
    :param z: b x 2 x n x n
    :return: Id: z: b x 2 x n x n
    """
    bs, _, _, dim_z = z.size()
    id = identity(dim_z).type(z.type()).to(z.device)
    id = id.unsqueeze(dim=0).repeat(bs, 1, 1, 1)
    return id


def is_hermitian(z: torch.Tensor):
    """
    Returns whether z is hermitian or not
    Z is hermitian iif Z = Z^*
    This is, Z is equal to the conjugate transpose of Z
    """
    return torch.allclose(z, conj_trans(z))


def to_hermitian(x: torch.Tensor):
    """Given a random matrix, alters the values to make it Hermitian.

    The conditions are:
    1 - real part is symmetric
    2 - imaginary diagonal must be zero
    3 - imaginary elements in the upper triangular part of the matrix must be of opposite sign than the
    elements in the lower triangular part

    :param x: b x * x 2 x n x n. Values can be randomly generated
    :return: b x * x 2 x n x n. All matrices are now Hermitian.
    """
    real_x = squared_to_symmetric(real(x))
    imag_x = squared_to_symmetric(imag(x))

    # imaginary diagonal must be zero and imaginary elements in the upper triangular part of the matrix must
    # be of opposite sign than the elements in the lower triangular part
    upper_triangular = torch.triu(imag_x, diagonal=1)       # this already zeros the diagonal
    lower_triangular = upper_triangular.transpose(-2, -1) * -1
    new_imag_x = lower_triangular + upper_triangular
    return stick(real_x, new_imag_x)


def to_compound_real_symmetric_from_hermitian(h: torch.Tensor):
    """
    Let Z = A + iB be a matrix with complex entries, where A, B are n x n matrices with real entries.
    If Z is Hermitian, then a 2n x 2n matrix can be created in the following form:
        M = [(A, -B),
             (B, A)]
    Since Z is Hermitian, then M is symmetric and with all real entries therefore we can calculate its eigenvalues.

    :param h: b x * x 2 x n x n. PRE: Each matrix in x must be Hermitian
    :return: m: b x * x 2n x 2n. Symmetric matrix with all real entries
    """
    real_h = real(h)
    imag_h = imag(h)
    a_and_minus_b = torch.cat((real_h, -imag_h), dim=-1)
    b_and_a = torch.cat((imag_h, real_h), dim=-1)
    m = torch.cat((a_and_minus_b, b_and_a), dim=-2)
    return m


def to_hermitian_from_compound_real_symmetric(m: torch.Tensor):
    """
    M is a 2n x 2n matrix built from a Hermitian matrix Z = A + iB, where
    M = [(A, -B),
         (B, A)]
    This function is the inverse operation of 'to_real_symmetric_from_hermitian' and returns Z

    :param m: b x * x 2n x 2n
    :return: z: b x * x 2 x n x n
    """
    a_on_top_of_b = torch.chunk(m, 2, dim=-1)[0]
    a, b = torch.chunk(a_on_top_of_b, 2, dim=-2)
    return stick(a, b)


def hermitian_eig(h: torch.Tensor):
    """
    Let Z = A + iB be a Hermitian matrix with complex entries, where A,B are n x n matrices with real entries.
    If Z is Hermitian, then a 2n x 2n matrix can be created in the following form:
        M = [(A, -B),
             (B, A)]
    Since Z is Hermitian, then M is symmetric and with all real entries therefore we can calculate its eigenvalues.

    Z has eigenvalues e1, e2,.. eN.
    M will have 2n eigenvalues, which will be: e1, e1, e2, e2,..., eN, eN.
    :param h: b x * x 2 x n x n. PRE: Each matrix must be Hermitian
    :return: eigenvalues of m, just as torch.symeig: 2n values
             eigenvectors of m, just as torch.symeig: 2n x 2n matrices
            eigenvalues of h: n values
    """
    m = to_compound_real_symmetric_from_hermitian(h)

    eigvalues, eigvectors = symeig(m)  # evalues are in ascending order
    h_eigvalues_index = torch.arange(start=0, end=eigvalues.size(-1), step=2, dtype=torch.long, device=eigvalues.device)
    h_eigvalues = eigvalues[:, h_eigvalues_index]

    return eigvalues, eigvectors, h_eigvalues


def hermitian_matrix_sqrt(h: torch.Tensor):
    """
    Calculates the sqrt for a Hermitian matrix
    :param h: b x * x 2 x n x n. PRE: Each matrix must be Hermitian
    :return: sqrt(h): b x * x 2 x n x n
    """
    sym_matrix = to_compound_real_symmetric_from_hermitian(h)
    mat_sqrt = matrix_sqrt(sym_matrix)
    hermitian_sqrt = to_hermitian_from_compound_real_symmetric(mat_sqrt)
    return hermitian_sqrt


def matrix_sqrt(y: torch.Tensor):
    """
    For a real symmetric matrix Y, the square root by diagonalization is:
    1) Y = SDS^-1
    2) D_sq = D^0.5 -> the sqrt of each entry in D
    3) Y_sqrt = S D_sq S^-1
    :param y: b x * x n x n. PRE: Each matrix must be symmetric
    :return: sqrt(y): b x * x n x n
    """
    eigvalues, eigvectors = symeig(y)
    d_sq = torch.diag_embed(torch.sqrt(eigvalues))
    return eigvectors.bmm(d_sq).bmm(eigvectors.inverse())
