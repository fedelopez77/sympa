from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SymmetricManifold
from sympa.manifolds import symmetric_math as smath

EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


class BoundedDomainManifold(SymmetricManifold):
    """
    Bounded domain Manifold.
    D_n = {z \in Sym(n, C) | Id - ẑz is positive definite}.
    This is, z models a point in the space D_n. z is a symmetric matrix of size nxn, with complex entries.

    This model generalizes the Poincare Disk model, which is when n = 1.
    """

    ndim = 1
    reversible = False
    name = "Bounded Domain"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, ndim=1):
        super().__init__(ndim=ndim)

    def r_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # TODO: validate the formula and implement it
        pass

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # TODO: see if this is the same than in the upper half space
        pass

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Currently, this function is only defined for n = 2

        :param x:
        :return:
        """
        v_entries = [(0, 0), (1, 1)]
        v = get_complex_vector_of_entries(x, v_entries)

        u_entries = [(0, 0), (0, 1), (1, 0), (1, 1)]
        u = get_complex_vector_of_entries(x, u_entries)

        max_norm = torch.max(v.norm(dim=1, keepdim=True), u.norm(dim=1, keepdim=True)).unsqueeze(dim=1)
        cond = max_norm >= 1
        max_norm = max_norm * (1 + EPS[x.dtype])
        real_x, imag_x = smath.real(x), smath.imag(x)
        projected_real_x = real_x / max_norm
        projected_imag_x = imag_x / max_norm

        final_real_x = torch.where(cond, projected_real_x, real_x)
        final_imag_x = torch.where(cond, projected_imag_x, imag_x)

        return smath.stick(final_real_x, final_imag_x)


def get_complex_vector_of_entries(v: torch.Tensor, entries: list):
    """
    :param v: tensor of b x 2 x n x n
    :param entries: list of tuples with the coordinate to take the real and imaginary part
    :return: tensor of b x 2k where k is the len of entries. Columns are real, imag, real, imag, ...
    """
    components = [v[:, :, i, j] for i, j in entries]
    return torch.cat(components, dim=1)


