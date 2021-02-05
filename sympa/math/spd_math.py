# Code taken from https://github.com/geoopt/geoopt/pull/153

from typing import Callable, Tuple
import torch
import torch.jit


@torch.jit.script
def trace(x: torch.Tensor) -> torch.Tensor:
    r"""self-implemented matrix trace, since `torch.trace` only support 2-d input.
    Parameters
    ----------
    x : torch.Tensor
        input matrix
    Returns
    -------
    torch.Tensor
        :math:`\operatorname{Tr}(x)`
    """
    return torch.diagonal(x, dim1=-2, dim2=-1).sum(-1)


def sym_funcm(
    x: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Apply function to symmetric matrix.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    func : Callable[[torch.Tensor], torch.Tensor]
        function to apply
    Returns
    -------
    torch.Tensor
        symmetric matrix with function applied to
    """
    e, v = torch.symeig(x, eigenvectors=True)
    return v @ torch.diag_embed(func(e)) @ v.transpose(-1, -2)


def sym_expm(x: torch.Tensor, using_native=False) -> torch.Tensor:
    r"""Symmetric matrix exponent.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    using_native : bool, optional
        if using native matrix exponent `torch.matrix_exp`, by default False
    Returns
    -------
    torch.Tensor
        :math:`\exp(x)`
    """
    # if using_native:
    #     return torch.matrix_exp(x)
    # else:
    return sym_funcm(x, torch.exp)


def sym_logm(x: torch.Tensor) -> torch.Tensor:
    r"""Symmetric matrix logarithm.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    Returns
    -------
    torch.Tensor
        :math:`\log(x)`
    """
    return sym_funcm(x, torch.log)


def sym_sqrtm(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix square root .
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    Returns
    -------
    torch.Tensor
        :math:`x^{1/2}`
    """
    return sym_funcm(x, torch.sqrt)


def sym_invm(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix inverse.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    Returns
    -------
    torch.Tensor
        :math:`x^{-1}`
    """
    return sym_funcm(x, torch.reciprocal)


def sym_inv_sqrtm1(x: torch.Tensor) -> torch.Tensor:
    """Symmetric matrix inverse square root.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    Returns
    -------
    torch.Tensor
        :math:`x^{-1/2}`
    """
    return sym_funcm(x, lambda tensor: torch.reciprocal(torch.sqrt(tensor)))


def sym_inv_sqrtm2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric matrix inverse square root, with square root return also.
    Parameters
    ----------
    x : torch.Tensor
        symmetric matrix
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        :math:`x^{-1/2}`, :math:`x^{1/2}`
    """
    e, v = torch.symeig(x, eigenvectors=True)
    sqrt_e = torch.sqrt(e)
    inv_sqrt_e = torch.reciprocal(sqrt_e)
    return (
        v @ torch.diag_embed(inv_sqrt_e) @ v.transpose(-1, -2),
        v @ torch.diag_embed(sqrt_e) @ v.transpose(-1, -2),
    )