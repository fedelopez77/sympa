import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SiegelManifold
from sympa.manifolds.metrics import MetricType
from sympa.math import csym_math as sm


class UpperHalfManifold(SiegelManifold):
    """
    Upper Half Manifold.
    H_n = {z \in Sym(n, C) | Imag(z) is positive definite}.
    This is, z models a point in the space H_n. z is a symmetric matrix of size nxn, with complex entries.

    This model generalizes the Upper Half plane, which is when n = 1.
    """

    ndim = 1
    reversible = False
    name = "Upper Half Space"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, dims=2, ndim=2, metric=MetricType.RIEMANNIAN):
        super().__init__(dims=dims, ndim=ndim, metric=metric)

    def egrad2rgrad(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        If you have a function f(z) on Hn, then the gradient is the  y * grad_eucl(f(z)) * y,
        where y is the imaginary part of z, and multiplication is just matrix multiplication.

        :param z: point on the manifold. Shape: (b, 2, n, n)
        :param u: gradient to be projected: Shape: same than z
        :return grad vector in the Riemannian manifold. Shape: same than z
        """
        real_grad, imag_grad = sm.real(u), sm.imag(u)
        y = sm.imag(z)
        real_grad = y.bmm(real_grad).bmm(y)
        imag_grad = y.bmm(imag_grad).bmm(y)
        return sm.stick(real_grad, imag_grad)

    def projx(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`z` on the manifold.

        In this space, we need to ensure that Y = Im(X) is positive definite.
        Since the matrix Y is symmetric, it is possible to diagonalize it.
        For a diagonal matrix the condition is just that all diagonal entries are positive, so we clamp the values
        that are <=0 in the diagonal to an epsilon, and then restore the matrix back into non-diagonal form using
        the base change matrix that was obtained from the diagonalization.

        Steps to project: Y = Im(z)
        1) Y = SDS^-1
        2) D_tilde = clamp(D, min=epsilon)
        3) Y_tilde = SD_tildeS^-1

        :param z: points to be projected: (b, 2, n, n)
        """
        z = super().projx(z)

        y = sm.imag(z)
        y_tilde, batchwise_mask = sm.positive_conjugate_projection(y)

        self.projected_points += len(z) - sum(batchwise_mask).item()

        return sm.stick(sm.real(z), y_tilde)

    def inner(self, z: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False) -> torch.Tensor:
        """
        Inner product for tangent vectors at point :math:`z`.
        For the upper half space model, the inner product at point z = x + iy of the vectors u, v
        it is (z, u, v are complex symmetric matrices):

        g_{z}(u, v) = tr[ y^-1 u y^-1 \ov{v} ]

        :param z: torch.Tensor point on the manifold: b x 2 x n x n
        :param u: torch.Tensor tangent vector at point :math:`z`: b x 2 x n x n
        :param v: Optional[torch.Tensor] tangent vector at point :math:`z`: b x 2 x n x n
        :param keepdim: bool keep the last dim?
        :return: torch.Tensor inner product (broadcastable): b x 2 x 1 x 1
        """
        if v is None:
            v = u
        inv_imag_z = torch.inverse(sm.imag(z))
        inv_imag_z = sm.stick(inv_imag_z, torch.zeros_like(inv_imag_z))

        res = sm.bmm3(inv_imag_z, u, inv_imag_z)
        res = sm.bmm(res, sm.conjugate(v))
        real_part = sm.trace(sm.real(res), keepdim=True)    # b x 1
        real_part = torch.unsqueeze(real_part, -1)          # b x 1 x 1
        return sm.stick(real_part, real_part)               # b x 2 x 1 x 1

    def _check_point_on_manifold(self, z: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        Util to check point lies on the manifold.
        For the Upper Half Space Model, that implies that Im(x) is positive definite.

        z is assumed to be one complex matrix with the shape 2 x ndim x ndim

        :param z: point on the manifold. (b, 2, n, n)
        :param atol: float, absolute tolerance as in :func:`torch.allclose`
        :param rtol: float, relative tolerance as in :func:`torch.allclose`
        :return bool, str or bool, None: check result and the reason of fail if any
        """
        if not self._check_matrices_are_symmetric(z, atol=atol, rtol=rtol):
            return False, "Matrices are not symmetric"

        imag_x = sm.imag(z.unsqueeze(0))
        ok = torch.det(imag_x) > 0
        if not ok:
            reason = "'x' determinant is not > 0"
        else:
            reason = None
        return ok, reason

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        """
        Random sampling on the manifold.

        The exact implementation depends on manifold and usually does not follow all
        assumptions about uniform measure, etc.
        """
        from_ = kwargs.get("from_", -0.001)
        to = kwargs.get("to", 0.001)
        dims = self.dims
        perturbation = sm.squared_to_symmetric(torch.Tensor(size[0], dims, dims).uniform_(from_, to))
        identity = torch.eye(dims).unsqueeze(0).repeat(size[0], 1, 1)
        imag_part = identity + perturbation

        real_part = sm.squared_to_symmetric(torch.Tensor(size[0], dims, dims).uniform_(from_, to))
        return sm.stick(real_part, imag_part).to(device=device, dtype=dtype)
