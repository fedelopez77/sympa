import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SymmetricManifold
from sympa.manifolds.upper_half import UpperHalfManifold
from sympa.manifolds.metric import Metric
from sympa.math import compsym_math as sm
from sympa.math.cayley_transform import cayley_transform, inverse_cayley_transform
from sympa.utils import get_logging

log = get_logging()


class BoundedDomainManifold(SymmetricManifold):
    """
    Bounded domain Manifold.
    D_n = {z \in Sym(n, C) | Id - ẑz is positive definite}.
    This is, z models a point in the space D_n. z is a symmetric matrix of size nxn, with complex entries.

    This model generalizes the Poincare Disk model, which is when n = 1.
    """

    ndim = 1
    reversible = False
    name = "BoundedDomain"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, dim=2, ndim=2, metric=Metric.RIEMANNIAN.value):
        super().__init__(dim=dim, ndim=ndim, metric=metric)

    def dist(self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        To compute the distance in the Bounded Domain Model (BDM) we need to map the elements to the
        Upper Half Space Model (UHSM) by means of the inverse Caley Transform, and then compute the distance
        in that domain.

        :param z1, z2: b x 2 x n x n: elements in the BDM
        :param keepdim:
        :return: distance between x and y
        """
        uhsm_z1 = inverse_cayley_transform(z1)
        uhsm_z2 = inverse_cayley_transform(z2)
        return super().dist(uhsm_z1, uhsm_z2)

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        If you have a function f(z) on Hn, then the gradient is the  A * grad_eucl(f(z)) * A,
        where A = (Id - \overline{Z}Z)
        :param z: point on the manifold. Shape: (b, 2, n, n)
        :param u: gradient to be projected: Shape: same than z
        :return grad vector in the Riemannian manifold. Shape: same than z
        """
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
        3) Z_tilde = Ŝ D_tilde S^*

        :param z: points to be projected: (b, 2, n, n)
        """
        z = super().projx(z)

        eigenvalues, s = self.takagi_factorization.factorize(z)
        eigenvalues_tilde = torch.clamp(eigenvalues, max=1 - sm.EPS[z.dtype])

        diag_tilde = torch.diag_embed(eigenvalues_tilde)
        diag_tilde = sm.stick(diag_tilde, torch.zeros_like(diag_tilde))

        z_tilde = sm.bmm3(sm.conjugate(s), diag_tilde, sm.conj_trans(s))

        # we do this so no operation is applied on the matrices that already belong to the space.
        # This prevents modifying values due to numerical instabilities/floating point ops
        batch_wise_mask = torch.all(eigenvalues < 1 - sm.EPS[z.dtype], dim=-1, keepdim=True)
        already_in_space_mask = batch_wise_mask.unsqueeze(-1).unsqueeze(-1).expand_as(z)

        self.projected_points += len(z) - sum(batch_wise_mask).item()

        return torch.where(already_in_space_mask, z, z_tilde)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        Util to check point lies on the manifold.
        For the Bounded Domain Model, that implies that Id - ẐZ is positive definite.

        x is assumed to be one complex matrix with the shape 2 x ndim x ndim

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        if not self._check_matrices_are_symmetric(x, atol=atol, rtol=rtol):
            return False, "Matrices are not symmetric"

        x = x.unsqueeze(0)
        id_minus_zz = get_id_minus_conjugate_z_times_z(x)
        ok = sm.is_hermitian(id_minus_zz)
        if not ok:
            reason = "'Id - ẐZ' is not hermitian (is not definite positive)"
        else:
            reason = None
        return ok, reason

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        """
        Random sampling on the manifold.

        The exact implementation depends on manifold and usually does not follow all
        assumptions about uniform measure, etc.
        """
        points = UpperHalfManifold(dim=self.dim).random(*size, **kwargs)
        return cayley_transform(points)


def get_id_minus_conjugate_z_times_z(z: torch.Tensor):
    """
    :param z: b x 2 x n x n
    :return: Id - \overline(z)z
    """
    identity = sm.identity_like(z)
    conj_z_z = sm.bmm(sm.conjugate(z), z)
    return sm.subtract(identity, conj_z_z)
