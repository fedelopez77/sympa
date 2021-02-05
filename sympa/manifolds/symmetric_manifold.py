from abc import ABC
from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold
from sympa.math import compsym_math as sm
from sympa.manifolds.metric import Metric
from sympa.math.cayley_transform import cayley_transform
from sympa.math.takagi_factorization import TakagiFactorization


def compute_finsler_metric_one(d: torch.Tensor) -> torch.Tensor:
    """
    Given d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
    the Finsler distance one (F_{1})is given by the summation of this values
    :param d: b x n: d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
    :return: b x 1: Finsler distance
    """
    res = torch.sum(d, dim=-1)
    return res


def compute_finsler_metric_infinity(d: torch.Tensor) -> torch.Tensor:
    """
    The Finsler distance infinity (F_{\infty}) is given by d_n = log((1 + r_n) / (1 - r_n)),
    with r_n the largest eigenvalues of the crossratio matrix,
    :param d: b x n: d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
            where r_0 the largest eigenvalue and r_{n-1} the smallest
    :return: b x 1: Finsler distance
    """
    res = d[:, 0]
    return res


def compute_finsler_metric_minimum(d: torch.Tensor) -> torch.Tensor:
    """
    Given d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
    the Finsler distance of minimum entropy (F_{min})is given by the summation of the weighted values
    \sum  2 * (n + 1 - i) * r_i
    :param d: b x n: d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
    :return: b x 1: Finsler distance
    """
    n = d.shape[-1]
    factor = 2
    weights = factor * (n + 1 - torch.arange(start=1, end=n + 1).unsqueeze(0))
    res = torch.sum(weights * d, dim=-1)
    return res


def compute_riemannian_metric(d: torch.Tensor) -> torch.Tensor:
    """
    Given d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
    the Riemannian distance is given by the summation of this values
    :param d: b x n: d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
    :return: b x 1: Riemannian distance
    """
    res = torch.sum(d**2, dim=-1)
    res = torch.sqrt(res)
    return res


class SymmetricManifold(Manifold, ABC):
    """
    Manifold to work on spaces S_n = {z in Sym(n, C)}. This is, z models a point in the space S. z is a symmetric
    matrix of size nxn, with complex entries.

    Abstract manifold that contains the common operations for manifolds of Symmetric matrices with complex entries
    """

    ndim = 1
    reversible = False
    name = "Symmetric Space"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, dim=2, ndim=2, metric=Metric.RIEMANNIAN.value):
        """
        Space of symmetric matrices of shape dim x dim

        :param dim: dimensions of the matrices
        :param ndim: number of dimensions of tensors. This parameter is not used in this class and
        it is kept for compatibility with Manifold base class
        """
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.takagi_factorization = TakagiFactorization(dim)
        self.projected_points = 0
        if metric == Metric.RIEMANNIAN.value:
            self.compute_metric = compute_riemannian_metric
        elif metric == Metric.FINSLER_ONE.value:
            self.compute_metric = compute_finsler_metric_one
        elif metric == Metric.FINSLER_INFINITY.value:
            self.compute_metric = compute_finsler_metric_infinity
        elif metric == Metric.FINSLER_MINIMUM.value:
            self.compute_metric = compute_finsler_metric_minimum
        elif metric == Metric.WEIGHTED_SUM.value:
            self.compute_metric = self.compute_weighted_sum
            self.weights = torch.nn.parameter.Parameter(torch.ones((1, self.dim)))
        else:
            raise ValueError(f"Unrecognized metric: {metric}")

    def dist(self, z1: torch.Tensor, z2: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        """
        This methods calculates the distance for the Upper Half Space Manifold (UHSM)
        It is implemented here since the way to calculate distances in the Bounded Domain Manifold requires mapping
        the points to the UHSM, and then applying this formula.

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

        # di = (1 + ri) / (1 - ri) # TODO: see if clamping only denom or whole result in case values are too large
        d = (1 + eigvalues) / (1 - eigvalues).clamp(min=eps)
        d = torch.log(d)
        res = self.compute_metric(d)
        return res

    def compute_weighted_sum(self, d: torch.Tensor) -> torch.Tensor:
        """
        Given d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
        we learn weights for each eigenvalue and apply a weighted average such that:
        distance = \sum  w_i * r_i
        :param d: b x n: d_i = log((1 + r_i) / (1 - r_i)), with r_i the eigenvalues of the crossratio matrix,
        :return: b x 1: Finsler distance
        """
        weights = torch.nn.functional.relu(self.weights)    # 1 x n
        res = weights * d
        res = torch.sum(res, dim=-1)
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
        ok = shape[-1] == self.dim and shape[-2] == self.dim
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(
                name, self, self.dim
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

    # I think I do not need to implement any of this methods!
    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # We might not need it
        pass

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # We might not need it
        pass

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # We might not need it
        pass

    def inner(self, x: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False) -> torch.Tensor:
        # We might not need it
        pass

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ) -> Union[Tuple[bool, Optional[str]], bool]:
        pass
