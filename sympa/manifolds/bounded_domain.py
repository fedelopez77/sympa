import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SymmetricManifold
from sympa.math import symmetric_math as sm
from sympa.math.caley_transform import inverse_caley_transform
from sympa.math.takagi_factorization import takagi_factorization


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

    def dist(self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        To compute the distance in the Bounded Domain Model (BDM) we need to map the elements to the
        Upper Half Space Model (UHSM) by means of the inverse Caley Transform, and then compute the distance
        in that domain.

        :param z1, z2: b x 2 x n x n: elements in the BDM
        :param keepdim:
        :return: distance between x and y
        """
        uhsm_z1 = inverse_caley_transform(z1)
        uhsm_z2 = inverse_caley_transform(z2)
        return super().dist(uhsm_z1, uhsm_z2)

    # def r_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     """
    #     R metric for the Bounded Domain Model:
    #     Given z1, z2 in D_n, R: D_n x D_n -> Mat(n, C)
    #         R(z1, z2) = (z1 - z2) (z1 - ẑ2^-1)^-1 (ẑ1^-1 - ẑ2^-1) (ẑ1^-1 - z2)^-1
    #     """
    #     x_conj_inverse = sm.inverse(sm.conjugate(x))
    #     y_conj_inverse = sm.inverse(sm.conjugate(y))
    #
    #     term_a = sm.subtract(x, y)
    #     term_b = sm.inverse(sm.subtract(x, y_conj_inverse))
    #     term_c = sm.subtract(x_conj_inverse, y_conj_inverse)
    #     term_d = sm.inverse(sm.subtract(x_conj_inverse, y))
    #
    #     res = sm.bmm(term_a, term_b)
    #     res = sm.bmm(res, term_c)
    #     res = sm.bmm(res, term_d)
    #     return res

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        If you have a function f(z) on Hn, then the gradient is the  A * grad_eucl(f(z)) * A,
        where A = (Id - \overline{Z}Z)

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
        a_times_grad_times_a = sm.bmm3(a, u, a)
        return a_times_grad_times_a

    def projx(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`z` on the manifold.

        In this space, we need to ensure that Y = Id - \overline(Z)Z is positive definite.

        Steps to project: Z complex symmetric matrix
        1) Z = SDS^-1
        2) D_tilde = clamp(D, max=1 - epsilon)
        3) Z_tilde = SD_tildeS^-1
        """
        # TODO: what if Z is not even symmetric? Impose symmetry?
        eigenvalues, s = takagi_factorization(z)
        eigenvalues_tilde = torch.clamp(eigenvalues, max=1 - sm.EPS[z.dtype])

        diag_tilde = torch.diag_embed(eigenvalues_tilde)
        diag_tilde = sm.stick(diag_tilde, torch.zeros_like(diag_tilde))

        z_tilde = sm.bmm3(sm.conjugate(s), diag_tilde, sm.conj_trans(s))
        return z_tilde


def get_id_minus_conjugate_z_times_z(z: torch.Tensor):
    """
    :param z: b x 2 x n x n
    :return: Id - \overline(z)z
    """
    identity = sm.identity_like(z)
    conj_z_z = sm.bmm(sm.conjugate(z), z)
    return sm.subtract(identity, conj_z_z)
