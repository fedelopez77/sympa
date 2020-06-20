from abc import ABC
from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold
from sympa.math import symmetric_math as sm
from sympa.math.caley_transform import caley_transform
from sympa.math.takagi_factorization import takagi_factorization


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

    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim

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
        real_z1, imag_z1 = sm.real(z1), sm.imag(z2)
        inv_sqrt_imag_z1 = sm.matrix_sqrt(imag_z1).inverse()
        inv_sqrt_imag_z1 = sm.stick(inv_sqrt_imag_z1, torch.zeros_like(inv_sqrt_imag_z1))
        z2_minus_real_z1 = sm.subtract(z2, sm.stick(real_z1, torch.zeros_like(real_z1)))
        z3 = sm.bmm3(inv_sqrt_imag_z1, z2_minus_real_z1, inv_sqrt_imag_z1)

        w = caley_transform(z3)

        eigvalues, eigvectors = takagi_factorization(w)

        # assert 1 >= eigvalues >= 0
        eps = sm.EPS[eigvalues.dtype]
        assert torch.all(eigvalues >= 0 - eps)
        assert torch.all(eigvalues <= 1 + eps)

        # ri = (1 + di) / (1 - di) # TODO: see if clamping only denom or whole result in case values are too large
        r = (1 + eigvalues) / (1 - eigvalues).clamp(min=eps)
        res = torch.sum(torch.log(r)**2, dim=-1)
        res = torch.sqrt(res)
        return res

    # def r_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    #     """
    #     The metric on S is governed by the function:
    #         R: S_n x S_n -> Mat(n, C)
    #
    #     The distance (geodesics) between two points x and y are a function of R.
    #     This is: d(x, y) = f(R(x, y))
    #
    #     :param x, y: Points in the space S_n. b x 2 x n x n
    #     :return: b x 2 x n x n
    #     """
    #     raise NotImplementedError

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Perform a retraction from point :math:`x` with given direction :math:`u`.

        Simple retraction: x_{t+1} = x_t + u
        Correct retraction: x_{t+1} = exp_map_{x_t}(u)

        :param x: point in the space
        :param u: usually the update is: -learning_rate * gradient
        :return:
        """
        # TODO: CHECK THIS!!!!!!!!!!
        # taken from geoopt.manifold.poincare: always assume u is scaled properly
        approx = x + u
        return self.projx(approx)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

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
        raise NotImplementedError

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

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[
        Tuple[bool, Optional[str]], bool]:
        # We might not need it
        pass

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[
        Tuple[bool, Optional[str]], bool]:
        # We might not need it
        pass

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        # We might not need it
        pass
