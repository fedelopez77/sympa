# Caley transform is used to map elements from the Upper Half Space Model to the Bounded Domain Model.
# The inverse Caley transform can be defined as well, and it maps elements from the Bounded Domain Model
# to the Upper Half Space Model.
# See https://en.wikipedia.org/wiki/Cayley_transform

from sympa.math import symmetric_math as sm
import torch


def caley_transform(z: torch.Tensor) -> torch.Tensor:
    """
    T(Z): Upper Half Space model -> Bounded Domain Model
    T(Z) = (Z - i Id)(Z + i Id)^-1

    :param z: b x 2 x n x n: PRE: z \in Upper Half Space Manifold
    :return: y \in Bounded Domain Manifold
    """
    identity = sm.identity_like(z)
    i_identity = sm.stick(sm.imag(identity), sm.real(identity))

    z_minus_id = sm.subtract(z, i_identity)
    inv_z_plus_id = sm.inverse(sm.add(z, i_identity))

    return sm.bmm(z_minus_id, inv_z_plus_id)


def inverse_caley_transform(z: torch.Tensor) -> torch.Tensor:
    """
    T^-1(Z): Bounded Domain Model -> Upper Half Space model
    T^-1(Z) = i (Id + Z)(Id - Z)^-1

    :param z: b x 2 x n x n: PRE: z \in Bounded Domain Manifold
    :return: y \in Upper Half Space Manifold
    """
    identity = sm.identity_like(z)

    i_z_plus_id = sm.multiply_by_i(sm.add(identity, z))
    inv_z_minus_id = sm.inverse(sm.subtract(identity, z))

    return sm.bmm(i_z_plus_id, inv_z_minus_id)
