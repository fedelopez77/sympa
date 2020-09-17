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
        """
        Space of symmetric matrices of shape ndim x ndim

        :param ndim: number of dimensions of the matrices
        """
        super().__init__()
        self.ndim = ndim
        self.projected_points = 0

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
        ok = shape[-1] == self.ndim and shape[-2] == self.ndim
        if not ok:
            reason = "'{}' on the {} requires more than {} dim".format(
                name, self, self.ndim
            )
        else:
            reason = None
        return ok, reason

    def _check_matrices_are_symmetric(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        :param x: point in the symmetric manifold of shape (2, ndims, ndims)
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
