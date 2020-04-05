from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold, ScalingInfo
from sympa.manifolds import SymmetricManifold
from sympa.manifolds import symmetric_math as smath


class UpperHalfManifold(SymmetricManifold):
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

    def __init__(self, ndim=1):
        super().__init__(ndim=ndim)

    def r_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Transform gradient computed using autodiff to the correct Riemannian gradient for the point :math:`x`.

        If you have a function f(z) on Hn, then the gradient is the  y * grad_eucl(f(z)) * y,
        where y is the imaginary part of z, and multiplication is just matrix multiplication.

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
        # TODO: CHECK THIS!!!!!!!!!!
        # TODO: If the gradient has also an imaginary part and a real part, this will fail. In that case it would be:
        # real_grad, imag_grad = smath.real(u), smath.imag(u)
        # y = smath.imag(x)
        # real_grad = y.bmm(real_grad).bmm(y)
        # imag_grad = y.bmm(imag_grad).bmm(y)
        # return smath.stick(real_grad, imag_grad)
        y = smath.imag(x)
        return y.bmm(u).bmm(y)

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point :math:`x` on the manifold.

        In this space, we need to ensure that Y = Im(X) is positive definite.
        Since the matrix Y is symmetric, it is possible to diagonalize it.
        For a diagonal matrix the condition is just that both diagonal entries are positive, so we clamp the values
        that are <=0 in the diagonal, and then restore the matrix back into non-diagonal form using the base change
        matrix that one obtained from the diagonalization.

        Steps to project: Y = Im(x)
        1) Y = SDS^-1
        2) D_tilde = clamp(D, min=0)
        3) Y_tilde = SD_tildeS^-1
        """
        s, eigenvalues = self.diagonalize(x)
        eigenvalues = torch.clamp(eigenvalues, min=0)
        d_tilde = torch.diag_embed(eigenvalues)
        y_tilde = s.bmm(d_tilde).bmm(s.inverse())
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
