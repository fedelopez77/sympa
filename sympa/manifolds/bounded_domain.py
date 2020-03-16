from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold, ScalingInfo


class BoundedDomainManifold(Manifold):

    ndim = 1
    reversible = False
    name = "Bounded Domain"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        pass

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    def logmap(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def transp(self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        pass

    def inner(self, x: torch.Tensor, u: torch.Tensor, v=None, *, keepdim=False) -> torch.Tensor:
        pass

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        pass

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[
        Tuple[bool, Optional[str]], bool]:
        pass

    def _check_vector_on_tangent(self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5) -> Union[
        Tuple[bool, Optional[str]], bool]:
        pass

    def random(self, *size, dtype=None, device=None, **kwargs) -> torch.Tensor:
        pass
