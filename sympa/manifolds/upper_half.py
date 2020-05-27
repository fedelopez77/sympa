import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SymmetricManifold
from sympa.math import symmetric_math as smath


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
        """
        R metric for the Upper half space model:
        Given z1, z2 in H_n, R: H_n x H_n -> Mat(n, C)
            R(z1, z2) = (z1 - z2) (z1 - ẑ2)^-1 (ẑ1 - ẑ2) (ẑ1 - z2)^-1
        """
        x_conj = smath.conjugate(x)
        y_conj = smath.conjugate(y)

        term_a = smath.subtract(x, y)
        term_b = smath.inverse(smath.subtract(x, y_conj))
        term_c = smath.subtract(x_conj, y_conj)
        term_d = smath.inverse(smath.subtract(x_conj, y))

        res = smath.bmm(term_a, term_b)
        res = smath.bmm(res, term_c)
        res = smath.bmm(res, term_d)
        return res

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
        For a diagonal matrix the condition is just that all diagonal entries are positive, so we clamp the values
        that are <=0 in the diagonal to an epsilon, and then restore the matrix back into non-diagonal form using
        the base change matrix that was obtained from the diagonalization.

        Steps to project: Y = Im(x)
        1) Y = SDS^-1
        2) D_tilde = clamp(D, min=epsilon)
        3) Y_tilde = SD_tildeS^-1
        """
        y = smath.imag(x)
        y_tilde = smath.positive_conjugate_projection(y)
        return smath.stick(smath.real(x), y_tilde)


def get_conjugate_of_r(x: torch.Tensor, y: torch.Tensor):
    """A = (ẑ1 - z2)(ẑ1 - ẑ2) ^ {-1}."""
    x_conj = smath.conjugate(x)
    y_conj = smath.conjugate(y)
    term_a = smath.subtract(x_conj, y)
    term_b = smath.inverse(smath.subtract(x_conj, y_conj))

    res = smath.bmm(term_a, term_b)
    return res
