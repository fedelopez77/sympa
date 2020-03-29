from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold, ScalingInfo
from sympa.manifolds import symmetric_math as smath


class UpperHalfSpaceManifold(Manifold):

    ndim = 1
    reversible = False
    name = "Upper Half Space"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        # TODO terminar de definir como carajo se calcula R, y esto en base a R
        r_metric = smath.r_metric(x, y)
        sqrt_r_metric = torch.pow(r_metric, 0.5)
        num = 1 + sqrt_r_metric
        denom = 1 - sqrt_r_metric
        log = torch.log(num / denom)
        sq_log = torch.pow(log, 2)




    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # taken from geoopt.manifold.poincare:
        # TODO: decidimos usar esta, ergo hacerla bien, pero mantener esta idea:
        # TODO: 1) sumarle u
        # TODO: 2) projectarlo dentro del espacio nuevamente
        # TODO: IDEM PARA BOUNDED DOMAIN!!!
        # always assume u is scaled properly
        """
        :param x: point in the space
        :param u: usually the update is: -learning_rate * gradient
        :return:
        """
        approx = x + u
        return math.project(approx, c=self.c, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # TODO: Anna: If you have a function f(z) on Hn, then the gradient is the  y*grad_eucl (f(z))*y,
        # TODO: Where y is the imaginary part of z, and multiplication is just matrix multiplication.
        pass

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # we can use egrad2rgrad
        pass

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`x` on the manifold.

        In this space, we need to ensure that Y = Im(X) is positive definite.
        Since the matrix Y is symmetric, it is possible to diagonalize it.
        For a diagonal matrix the condition is just that both diagonal entries are positive, so we clamp the values
        that are <=0 in the diagonal, and then restore the matrix back into non-diagonal form using the base change
        matrix that one obtained from the diagonalization.

        Procedure to project: Y = Im(x)
        1 - Y = SDS^-1
        2 - D_tilde = clamp(D, min=0)
        3 - Y_tilde = SD_tildeS^-1
        """
        s, eigenvalues = self.diagonalize(x)
        eigenvalues = torch.clamp(eigenvalues, min=0)
        d_tilde = torch.diag_embed(eigenvalues)
        y_tilde = torch.bmm(torch.bmm(s, d_tilde), torch.inverse(s))
        return smath.stick(smath.real(x), y_tilde)

    def diagonalize(self, y: torch.Tensor):
        """
        Y = Im(X)   Y is a squared symmetric matrix, then Y can be decomposed (diagonalized) as Y = SDS^-1
        where S is the matrix composed by the eigenvectors of Y (S = [v_1, v_2, ..., v_n] where v_i are the
        eigenvectors), D is the diagonal matrix constructed from the corresponding eigenvalues, and S^-1 is the
        matrix inverse of S.

        This function could be implemented as:
        Pre: y = smath.imag(x)                                                      # b x n x n
        evalues, evectors = torch.symeig(y, eigenvectors=True)                      # evalues are in ascending order
        return evectors, torch.diag_embed(evalues), torch.inverse(evectors)

        but since I don't need everything I do not return the eigenvalues in a diagonal matrix form.

        :param y: b x n x n
        :return: eigenvalues, eigenvectors
        """
        eigenvalues, eigenvectors = torch.symeig(y, eigenvectors=True)                  # evalues are in ascending order
        return eigenvalues, eigenvectors

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
