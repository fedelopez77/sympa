from typing import Union, Tuple, Optional
import torch
from geoopt.manifolds.base import Manifold, ScalingInfo
from sympa.manifolds import complex_math


class UpperHalfSpaceManifold(Manifold):

    ndim = 1
    reversible = False
    name = "Upper Half Space"
    __scaling__ = Manifold.__scaling__.copy()

    def __init__(self, ndim=1):
        super().__init__()
        self.ndim = ndim
        # TODO calculate indexes of diagonal and store them to calculate trace later

    def dist(self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False) -> torch.Tensor:
        r_metric = complex_math.r_metric(x, y)
        sqrt_r_metric = torch.pow(r_metric, 0.5)
        num = 1 + sqrt_r_metric
        denom = 1 - sqrt_r_metric
        log = torch.log(num / denom)
        sq_log = torch.pow(log, 2)




    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # taken from geoopt.manifold.poincare
        # always assume u is scaled properly
        approx = x + u
        return math.project(approx, c=self.c, dim=dim)

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

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # we can use egrad2rgrad
        pass

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # need to define which kind of product they mean
        # diria que un hadamard
        pass

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        # esto es donde impongo lo que me mando en el mail anterior
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

    # TODO If I need other operations such as addition, hadamard product, etc, it should be written here as well