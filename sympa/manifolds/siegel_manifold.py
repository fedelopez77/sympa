from abc import ABC
from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold
from sympa.math import csym_math as sm
from sympa.manifolds.metrics import Metric, MetricType
from sympa.math.cayley_transform import cayley_transform
from sympa.math.takagi_factorization import TakagiFactorization


class SiegelManifold(Manifold, ABC):
    """
    Manifold to work on Siegel spaces.
    The implementation is aimed to work with realization of the Siegel space as
    spaces of complex symmetric matrices.
    These are of the shape: b x 2 x dim x dim
    """

    ndim = 1
    reversible = False
    name = "Siegel Space"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, dims=2, ndim=2, metric=MetricType.RIEMANNIAN, use_xitorch=False):
        """
        Space of symmetric matrices of shape dims x dims

        :param dims: dimensions of the matrices
        :param ndim: number of dimensions of tensors. This parameter is not used in this class and
        it is kept for compatibility with Manifold base class
        :param metric: metric function to apply for measuring distances. It has to be a Metric value
        :param use_xitorch: if True it uses sm.xitorch_symeig. If False it uses sm.symeig.
        """
        super().__init__()
        self.dims = dims
        self.ndim = ndim
        self.projected_points = 0
        self.takagi_factorization = TakagiFactorization(dims, use_xitorch=use_xitorch)
        self.metric = Metric.get(metric, self.dims)

    def dist(self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        Calculates the distance for the Upper Half Space Manifold (UHSM)
        It is implemented here since the way to calculate distances in the Bounded Domain Manifold
        requires mapping the points to the UHSM, and then applying this formula.

        :param z1, z2: b x 2 x n x n: elements in the UHSM
        :param keepdim:
        :return: distance between x and y in the UHSM
        """
        # with Z1 = X + iY, define Z3 = sqrt(Y)^-1 (Z2 - X) sqrt(Y)^-1
        real_z1, imag_z1 = sm.real(z1), sm.imag(z1)
        inv_sqrt_imag_z1 = sm.matrix_sqrt(imag_z1).inverse()
        inv_sqrt_imag_z1 = sm.stick(inv_sqrt_imag_z1, torch.zeros_like(inv_sqrt_imag_z1))
        z2_minus_real_z1 = sm.subtract(z2, sm.stick(real_z1, torch.zeros_like(real_z1)))
        z3 = sm.bmm3(inv_sqrt_imag_z1, z2_minus_real_z1, inv_sqrt_imag_z1)

        w = cayley_transform(z3)

        eigvalues, eigvectors = self.takagi_factorization.factorize(w)

        # assert 1 >= eigvalues >= 0
        eps = sm.EPS[eigvalues.dtype]
        assert torch.all(eigvalues >= 0 - eps), f"Eigenvalues: {eigvalues}"
        assert torch.all(eigvalues <= 1.01), f"Eigenvalues: {eigvalues}"

        # vi = (1 + di) / (1 - di)
        v = (1 + eigvalues) / (1 - eigvalues).clamp(min=eps)
        v = torch.log(v)
        res = self.metric.compute_metric(v)
        return res

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Perform a retraction from point :math:`x` with given direction :math:`u`.

        Simple retraction: x_{t+1} = x_t + u
        Correct retraction: x_{t+1} = exp_map_{x_t}(u)

        :param x: point in the space
        :param u: usually the update is: -learning_rate * gradient
        :return:
        """
        # taken from geoopt.manifold.poincare: always assume u is scaled properly
        approx = x + u
        return self.projx(approx)

    def _check_shape(self, shape: Tuple[int], name: str) -> Union[Tuple[bool, Optional[str]], bool]:
        """
        Util to check shape.

        Exhaustive implementation for checking if a given point has valid dimension size,
        shape, etc. It should return boolean and a reason of failure if check is not passed

        This function is overridden from its original implementation to work with spaces of
        matrices

        Parameters
        ----------
        shape : Tuple[int]
            shape of point on the manifold
        name : str
            name to be present in errors

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        ok = shape[-1] == self.dims and shape[-2] == self.dims
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(
                name, self, self.dims
            )
        else:
            reason = None
        return ok, reason

    def _check_matrices_are_symmetric(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        :param x: point in the symmetric manifold of shape (2, dim, dim)
        :param atol:
        :param rtol:
        :return: True if real and imaginary parts of the point x are symmetric matrices,
        False otherwise
        """
        return sm.is_complex_symmetric(x.unsqueeze(0), atol, rtol)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`x` on the manifold.
        Ensures that the point x is made of symmetric matrices

        :param x: points to be projected: (b, 2, n, n)
        """
        return sm.to_symmetric(x)

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.egrad2rgrad(x, u)

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        r"""
        Perform vector transport :math:`\mathfrak{T}_{x\to y}(v)`.

        The formula for the parallel transport is very complicated so we approximate it
        with the Euclidean parallel transport, which is just returning the tangent vector v

        :param x: torch.Tensor. Start point on the manifold
        :param y: torch.Tensor. Target point on the manifold
        :param v: torch.Tensor. Tangent vector at point :math:`x`
        :return: torch.Tensor. Transported tensor
        """
        return v

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # We might not need it
        pass

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # We might not need it
        pass

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        pass
