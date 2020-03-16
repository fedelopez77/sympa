import torch


def r_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Implement R metric as:
    R(z_1, z_2) = (z_1 - z_2) (z_1 - ẑ_2)^{-1} (ẑ_1 - z_2) (ẑ_1 - ??????
    """
    # TODO
    pass


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
    reurn torch.eye(ndim)
