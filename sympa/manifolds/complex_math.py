# All complex_ operations assume that the tensors are of the shape (b, 2, n, n), which represent a matrix with complex
# entries
# (or cmat in case that I also need complex operations)

import torch


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res.to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def real(x: torch.Tensor):
    return x[:, 0]


def imag(x: torch.Tensor):
    return x[:, 1]


def stick(real_part: torch.Tensor, imag_part: torch.Tensor):
    """Returns a complex tensor made of the real and imaginary parts"""
    return torch.stack((real_part, imag_part), dim=1)


def conjugate(x: torch.Tensor):
    return stick(real(x), -imag(x))


def abs(x: torch.Tensor):
    result = torch.sqrt(real(x) ** 2 + imag(x) ** 2)
    return result


def complex_add(x: torch.Tensor, y: torch.Tensor):
    return x + y


def complex_sub(x: torch.Tensor, y: torch.Tensor):
    return x - y


def complex_pow(x: torch.Tensor, exponent):
    """
    x = a + ib = r (cosθ + i sinθ) where r^2 = a^2 + b^2 and tanθ = b / a.
    Then: (a + ib)^n = r^n (cos(nθ) + i sin(nθ)).
    """
    r = abs(x)
    r = torch.pow(r, exponent)
    tita = artanh(imag(x) / real(x))
    tita = tita * exponent
    real_part = r * torch.cos(tita)
    imag_part = r * torch.sin(tita)
    return stick(real_part, imag_part)


def complex_bmm(x: torch.Tensor, y: torch.Tensor):
    """
    x = a + ib; y = c + id
    xy = (a + ib)(c + id)
       = ac + iad + ibc - bd
       = (ac - bd) + i(ad + bc)

    :param x, y: tensors of b x 2 x n x n
    """
    real_x, imag_x = real(x), imag(x)
    real_y, imag_y = real(y), imag(y)
    ac = real_x.bmm(real_y)
    bd = imag_x.bmm(imag_y)
    ad = real_x.bmm(imag_y)
    bc = imag_y.bmm(real_y)
    out_real = ac - bd
    out_imag = ad + bc
    return stick(out_real, out_imag)


def repr(x: torch.Tensor):
    batch_size, _, n, _ = x.size()
    real_x, imag_x = real(x), imag(x)
    result = []
    for b_i in range(batch_size):
        real_item, imag_item = real_x[b_i], imag_x[b_i]
        rows = []
        for i in range(n):
            row = [f"{real_item[i][j]:.4f}+{imag_item[i][j]:.4f}j" for j in range(n)]
            rows.append("    ".join(row))
        result.append("\n".join(rows))
        result.append("")
    return "\n".join(result)


def trace(x: torch.Tensor, diagonal_index=None) -> torch.Tensor:
    """
    :param x: b x 2 x ndims x ndims
    :return:
    """

    # Mepa que con usar esta funcion alcanza: https://pytorch.org/docs/stable/torch.html#torch.diagonal


    if diagonal_index is None:
        ndims = x.size(-1)
        diagonal_index = torch.eye(ndims)




    # TODO recibo la diagonal_index o la calculo, y luego calculo la traza de todo esto
    # Idea: calculo la traza real, calculo la traza imaginaria, y luego devuelvo a modo de tupla o nro complejo, ya que
    # es la suma de los n nros complejos que estan en la diagonal. Ver si no hay funciones ya para obtener la diagonal
    pass


def diagonal_index(ndim: int) -> torch.Tensor:
    # TODO calcular los indices de la diagonal de un tensor de ndim x ndim y ver si esto sirve asi para indexar en la funcion trace
    return torch.eye(ndim)
