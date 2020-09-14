import torch
from geoopt.manifolds.base import Manifold
from sympa.manifolds import SymmetricManifold
from sympa.math import symmetric_math as sm


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
        y = sm.imag(x)
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
        y = sm.imag(x)
        y_tilde = sm.positive_conjugate_projection(y)
        return sm.stick(sm.real(x), y_tilde)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        """
        Util to check point lies on the manifold.
        For the Upper Half Space Model, that implies that Im(x) is positive definite.

        x is assumed to be one complex matrix with the shape 2 x ndim x ndim

        Parameters
        ----------
        x torch.Tensor
            point on the manifold
        atol: float
            absolute tolerance as in :func:`numpy.allclose`
        rtol: float
            relative tolerance as in :func:`numpy.allclose`

        Returns
        -------
        bool, str or None
            check result and the reason of fail if any
        """
        imag_x = sm.imag(x.unsqueeze(0))
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
        points = generate_matrix_in_upper_half_space(size[0], self.ndim, **kwargs)
        return points.to(device=device, dtype=dtype)


def generate_matrix_in_upper_half_space(points: int, dims: int, **kwargs):
    """
    Generates 'points' matrices in the Upper Half Space Model of 'dims' dimensions.
    This matrices already belong to the UHSM so they do not need to be projected.

    The imaginary part is generated in a way that if the matrix A is 2x2 with [[a1, b1], [b1, a2]]
     - a_i > 0
     - a1 * a2 - b1^2 > 0
    So the determinant of A is > 0
    For this, we impose: |b_i| < sqrt(a1 * a2) / 2

    The real part can be any symmetric matrix

    :param points: amount of matrices that will be in the batch
    :param dims: number of dimensions of the matrix
    :param epsilon: a_i values will be >= epsilon
    :param top: samples numbers from a Uniform distribution U(epsilon, top)
    :return: tensor of points x 2 x dims x dims
    """
    epsilon = kwargs.pop("epsilon", 0.001)
    top = kwargs.pop("top", 0.01)

    imag = torch.Tensor(points, dims, dims).uniform_(epsilon, top)
    for p in range(points):
        for i in range(dims):
            for j in range(i + 1, dims):
                a_i = imag[p, i, i]
                a_j = imag[p, j, j]
                threshold = torch.sqrt(a_i * a_j) / 2
                b_ij = imag[p, i, j]
                if torch.abs(b_ij) < threshold:         # condition is satisfied
                    imag[p, j, i] = b_ij                # makes matrix symmetric
                else:
                    abs_max_val = threshold * (1 - epsilon)
                    imag[p, i, j] = imag[p, j, i] = torch.Tensor(1).uniform_(-abs_max_val, abs_max_val)

        # checks that all the "sub" determinants in the matrix are > 0
        for k in range(1, dims + 1):
            sub_det = torch.det(imag[p, :k, :k])
            assert sub_det > 0

    real = sm.squared_to_symmetric(torch.Tensor(points, dims, dims).uniform_(-top, top))
    return sm.stick(real, imag)
