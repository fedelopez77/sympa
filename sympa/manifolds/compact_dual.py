import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SiegelManifold
from sympa.manifolds.metrics import MetricType
from sympa.math import csym_math as sm


class CompactDualManifold(SiegelManifold):
    """
    Compact Dual Manifold.
    M_n = {z \in Sym(n, C) }.
    This is, z models a point in the space M_n. z is a symmetric matrix of size nxn, with complex entries.
    """

    ndim = 1
    reversible = False
    name = "Compact Dual"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, dims=2, ndim=2, metric=MetricType.RIEMANNIAN.value):
        # Sets use_xitorch to True due to instabilities in the gradient
        super().__init__(dims=dims, ndim=ndim, metric=metric, use_xitorch=True)

    def dist(self, w: torch.Tensor, x: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Given  W, X in the compact dual:
            1 - TakagiFact(W) -> W = ÛPU*
            2 - U unitary, P diagonal, Û: U conjugate, U*: U conjugate transpose
            3 - Define A = (Id + P^2)^(-1/2)
            4 - Define M = [(A  -AP), (AP  A)] * [(U^t  0), (0  U)]
            5 - MW = 0 by construction. Y = MX implies
                Y = [(A  -AP), (AP  A)] * [(U^t  0), (0  U)] * X
                Y = [(A  -AP), (AP  A)] * U^tXU        Lets call Q = U^tXU
                Y = (AQ - AP) (APQ + A)^-1
            6 - TakagiFact(Y) = ŜDS*
            7 - Distance = sqrt[ sum ( arctan(d_k)^2 ) ]  with d_k the diagonal entries of D

        :param w, x: b x 2 x n x n: elements in the Compact Dual
        :param keepdim:
        :return: distance between w and x in the compact dual
        """
        p, u = self.takagi_factorization.factorize(w)  # p: b x n, u: b x 2 x n x n

        # Define A: since (Id + P^2) is diagonal, taking the matrix sqrt is just the sqrt of the entries
        # Moreover, then taking the inverse of that is taking the inverse of the entries.
        a = 1 + p**2
        a = 1 / torch.sqrt(a)
        a = sm.diag_embed(a)
        p = sm.diag_embed(p)

        q = sm.bmm3(sm.transpose(u), x, u)
        ap = sm.bmm(a, p)
        aq_minus_ap = sm.subtract(sm.bmm(a, q), ap)
        apq_plus_a_inv = sm.add(sm.bmm(ap, q), a)
        apq_plus_a_inv = sm.inverse(apq_plus_a_inv)
        y = sm.bmm(aq_minus_ap, apq_plus_a_inv)                     # b x 2 x n x n

        d, s = self.takagi_factorization.factorize(y)   # d = b x n

        d = torch.atan(d)
        dist = self.compute_metric(d)
        return dist

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        If you have a function f(z) on Mn, then the Riemannian gradient is
            grad_R(f(z)) = (Id + ẑz) * grad_E(f(z)) * (Id + zẑ)

        :param z: point on the manifold. Shape: (b, 2, n, n)
        :param u: gradient to be projected: Shape: same than z
        :return grad vector in the Riemannian manifold. Shape: same than z
        """
        id = sm.identity_like(z)
        conjz = sm.conjugate(z)
        id_plus_conjz_z = id + sm.bmm(conjz, z)
        id_plus_z_conjz = id + sm.bmm(z, conjz)
        riem_grad = sm.bmm3(id_plus_conjz_z, u, id_plus_z_conjz)
        return riem_grad

    def _check_point_on_manifold(self, z: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        Util to check point lies on the manifold.

        :param z: point on the manifold. (b, 2, n, n)
        :param atol: float, absolute tolerance as in :func:`torch.allclose`
        :param rtol: float, relative tolerance as in :func:`torch.allclose`
        :return bool, str or bool, None: check result and the reason of fail if any
        """
        if not self._check_matrices_are_symmetric(z, atol=atol, rtol=rtol):
            return False, "Matrices are not symmetric"
        return True, None

    def inner(self, z: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False) -> torch.Tensor:
        raise NotImplementedError()

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        """
        Random sampling on the manifold.
        "Base" points are symmetric perturbations around the zero matrix, just like
        in the Bounded Domain
        """
        from sympa.manifolds import BoundedDomainManifold
        return BoundedDomainManifold(dims=self.dims).random(size[0], **kwargs)
