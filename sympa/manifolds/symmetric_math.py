# Definition of operations for points represented as symmetric matrices with complex entries.
# All sym_ operations assume that tensors are of the shape (b, 2, n, n), which represent a matrix with complex
# entries.
# b: batch size
# 2: element in position 0 represents real part of the complex entry
#    element in position 1 represents imaginary part of the complex entry
# n: dimensions of the matrix

import torch
from geoopt.manifolds.poincare.math import artanh


def real(x: torch.Tensor):
    return x[:, 0]


def imag(x: torch.Tensor):
    return x[:, 1]


def stick(real_part: torch.Tensor, imag_part: torch.Tensor):
    """Returns a symmetric complex tensor made of the real and imaginary parts"""
    return torch.stack((real_part, imag_part), dim=1)


def sym_conjugate(x: torch.Tensor):
    return stick(real(x), -imag(x))


def sym_abs(x: torch.Tensor):
    result = torch.sqrt(real(x) ** 2 + imag(x) ** 2)
    return result


def sym_add(x: torch.Tensor, y: torch.Tensor):
    return x + y


def sym_sub(x: torch.Tensor, y: torch.Tensor):
    return x - y


def sym_pow(x: torch.Tensor, exponent):
    """
    x = a + ib = r (cosθ + i sinθ) where r^2 = a^2 + b^2 and tanθ = b / a.
    Then: (a + ib)^n = r^n (cos(nθ) + i sin(nθ)).
    """
    r = sym_abs(x)
    r = torch.pow(r, exponent)
    tita = artanh(imag(x) / real(x))
    tita = tita * exponent
    real_part = r * torch.cos(tita)
    imag_part = r * torch.sin(tita)
    return stick(real_part, imag_part)


def sym_bmm(x: torch.Tensor, y: torch.Tensor):
    """
    x = a + ib; y = c + id
    xy = (a + ib)(c + id)
       = ac + iad + ibc - bd
       = (ac - bd) + i(ad + bc)

    :param x, y: tensors of b x 2 x n x n
    """
    real_x, imag_x = real(x), imag(x)
    real_y, imag_y = real(y), imag(y)
    ac = real_x.bmm(real_y)
    bd = imag_x.bmm(imag_y)
    ad = real_x.bmm(imag_y)
    bc = imag_y.bmm(real_y)
    out_real = ac - bd
    out_imag = ad + bc
    return stick(out_real, out_imag)


def sym_make_symmetric(x: torch.Tensor):
    """
    Copies the values on the upper triangular to the lower triangular in order to make it symmetric
    :param x: b x 2 x n x n
    :return:
    """
    real_sym = _make_symmetric(real(x))
    imag_sym = _make_symmetric(imag(x))
    return stick(real_sym, imag_sym)


def _make_symmetric(x: torch.Tensor):
    """
    Copies the values on the upper triangular to the lower triangular in order to make it symmetric

    Alternative: M + M.transpose() is always symmetric

    :param x: b x n x n
    :return:
    """
    # takes the upper triangular and transpose it
    lower_triangular = torch.triu(x, diagonal=1).transpose(1, 2)
    upper_triangular = torch.triu(x)
    return lower_triangular + upper_triangular


def sym_repr(x: torch.Tensor):
    batch_size, _, n, _ = x.size()
    real_x, imag_x = real(x), imag(x)
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


def sym_inverse(x: torch.Tensor):
    """
    It calculates the inverse of a matrix with complex entries according to the algorithm explained here:
    "Method to Calculate the Inverse of a Complex Matrix using Real Matrix Inversion" by Andreas Falkenberg
    https://pdfs.semanticscholar.org/f278/b548b5121fd0d09c2e589439b97fad16ece3.pdf

    Given a squared matrix with complex entries Z, such that Z = A + iC where A,C are squared matrices with real
    entries, the algorithm returns M = U + iV, where M = Inverse(Z) and U,V are squared matrices with real entries
    Steps:
        R = inverse(A) * C
        U = inverse(C * R + A)
        V = - (R * U)

    * denotes matrix multiplication
    """
    a, c = real(x), imag(x)

    r = torch.inverse(a).bmm(c)
    u = torch.inverse(c.bmm(r) + a)
    v = (r.bmm(u)) * -1
    return stick(u, v)
