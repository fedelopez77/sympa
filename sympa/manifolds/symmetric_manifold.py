from abc import ABC
from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import symmetric_math as smath


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

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        # TODO terminar de definir como carajo se calcula R, y esto en base a R
        r_metric = self.r_metric(x, y)
        sqrt_r_metric = torch.pow(r_metric, 0.5)
        num = 1 + sqrt_r_metric
        denom = 1 - sqrt_r_metric
        log = torch.log(num / denom)
        sq_log = torch.pow(log, 2)


    def r_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        The metric on S is governed by the function:
            R: S_n x S_n -> Mat(n, C)

        The distance (geodesics) between two points x and y are a function of R.
        This is: d(x, y) = f(R(x, y))

        :param x, y: Points in the space S_n
        :return: Matrix of nxn
        """
        raise NotImplementedError

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
