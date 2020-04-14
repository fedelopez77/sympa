from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SymmetricManifold
from sympa.manifolds import symmetric_math as smath


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

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        If you have a function f(z) on Hn, then the gradient is the  A * grad_eucl(f(z)) * A,
        where A = (Id - \overline{Z}Z), and multiplication is just matrix multiplication.

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        u torch.Tensor
            gradient to be projected

        Returns
        -------
        torch.Tensor
            grad vector in the Riemannian manifold
        """
        # TODO: Check this!!!!!!!!!!!!!!!!!!
        # Assuming that the gradient is a tensor of b x 2 x n x n and it has a real and an imaginary part:
        a = get_id_minus_conjugate_z_times_z(z)
        a_times_grad = smath.sym_bmm(a, u)
        a_times_grad_times_a = smath(a_times_grad, a)
        return a_times_grad_times_a

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
        max_norm = max_norm * (1 + smath.EPS[x.dtype])
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


def get_id_minus_conjugate_z_times_z(z: torch.Tensor):
    """
    :param z: b x 2 x n x n
    :return: Id - \overline(z)z
    """
    identity = smath.sym_identity_from_tensor(z)
    conj_z = smath.sym_conjugate(z)
    conj_z_z = smath.sym_bmm(conj_z, z)
    return smath.sym_sub(identity, conj_z_z)
