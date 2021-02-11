import abc
import torch
import torch.nn as nn
from sympa import config
import geoopt as gt
from geoopt.manifolds.symmetric_positive_definite import SymmetricPositiveDefinite
from sympa.manifolds.metrics import MetricType
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold, CompactDualManifold


class Embeddings(nn.Module, abc.ABC):
    """Abstract Embedding layer that operates with embeddings as a Manifold parameter"""

    def __init__(self, num_embeddings, dims, manifold, _embeds):
        """
        :param num_embeddings: number of elements in the table
        :param dims: dimensionality of embeddings.
        If it is vector embeddings, it is the amount of dimensions in the vector.
        If it is Matrix embeddings, it is the last dimension on the matrix
        :param manifold: Instance of Geoopt Manifold
        :param _embeds: tensor with initial value for embeddings
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        self.manifold = manifold
        self.embeds = gt.ManifoldParameter(_embeds, manifold=self.manifold)

    def forward(self, input_index):
        """
        :param input_index: tensor of b x 1
        :return: Embeddings of b x * x n
        """
        return self.embeds[input_index]

    def proj_embeds(self):
        """Projects embeddings back into the manifold"""
        with torch.no_grad():
            self.embeds.data = self.manifold.projx(self.embeds.data)

    def check_all_points(self):
        for i in range(len(self.embeds)):
            point = self.embeds.data[i]
            ok, reason = self.manifold.check_point_on_manifold(point, explain=True)
            if not ok:
                return False, point, reason
        return True, None, None

    @abc.abstractmethod
    def norm(self):
        pass


class MatrixEmbeddings(Embeddings):

    def __init__(self, num_embeddings, dims, manifold):
        """
        Represents Embeddings of Matrices.
        The shape of the embedding table for complex matrices is: num_embeddings x 2 x dims x dims
        The second dimension is 2, since the element 0 represents the real part,
        and the element 1 represents the imaginary part of the matrices.
        For regular matrices (SPD in particular) the shape is: num_embeddings x embedding_dim x embedding_dim

        :param num_embeddings: number of elements in the table
        :param dims: dimensionality of matrix embeddings.
        :param manifold: SymmetricPositiveDefinite, UpperHalfManifold, BoundedDomainManifold
        """
        _embeds = manifold.random(num_embeddings, dims, dims, from_=-config.INIT_EPS, to=config.INIT_EPS)

        if isinstance(manifold, SymmetricPositiveDefinite):   # Scales and moves the SPD embeddings near the identity
            _embeds *= config.INIT_EPS
            _embeds += torch.diag_embed(torch.ones(num_embeddings, dims))

        super().__init__(num_embeddings, dims, manifold, _embeds)

    def norm(self):
        points = self.embeds.data
        points = points.reshape(len(points), -1)
        return points.norm(dim=-1)


class VectorEmbeddings(Embeddings):
    def __init__(self, num_embeddings, dims, manifold):
        init_eps = config.INIT_EPS
        _embeds = torch.Tensor(num_embeddings, dims).uniform_(-init_eps, init_eps)
        super().__init__(num_embeddings, dims, manifold, _embeds)
        self.proj_embeds()

    def norm(self):
        return self.embeds.data.norm(dim=-1)


class EmbeddingsFactory:
    @classmethod
    def get_embeddings(cls, name: str, num_points: int, dims: int, manifold):
        embed_table = cls._get_table(name)
        return embed_table(num_embeddings=num_points, dims=dims, manifold=manifold)

    @classmethod
    def _get_table(cls, model_name: str):
        if model_name in {"upper", "bounded", "dual", "spd"}:
            return MatrixEmbeddings
        if model_name in {"euclidean", "poincare", "lorentz", "sphere", "prod-hysph", "prod-hyhy", "prod-hyeu"}:
            return VectorEmbeddings
        raise ValueError(f"Unrecognized embedding model: {model_name}")


def get_prod_hysph_manifold(dims):
    poincare = gt.PoincareBall()
    sphere = gt.Sphere()
    return gt.ProductManifold((poincare, dims // 2), (sphere, dims // 2))


def get_prod_hyhy_manifold(dims):
    poincare = gt.PoincareBall()
    return gt.ProductManifold((poincare, dims // 2), (poincare, dims // 2))


def get_prod_hyeu_manifold(dims):
    poincare = gt.PoincareBall()
    euclidean = gt.Euclidean(1)
    return gt.ProductManifold((poincare, dims // 2), (euclidean, dims // 2))


class ManifoldFactory:

    geoopt_manifolds = {
        "euclidean": lambda dims: gt.Euclidean(1),
        "poincare": lambda dims: gt.PoincareBall(),
        "lorentz": lambda dims: gt.Lorentz(),
        "sphere": lambda dims: gt.Sphere(),
        "prod-hysph": get_prod_hysph_manifold,
        "prod-hyhy": get_prod_hyhy_manifold,
        "prod-hyeu": get_prod_hyeu_manifold,
        "spd": lambda dims: SymmetricPositiveDefinite(),
    }

    sympa_manifolds = {
        "upper": UpperHalfManifold,
        "bounded": BoundedDomainManifold,
        "dual": CompactDualManifold
    }

    @classmethod
    def get_manifold(cls, manifold_name, metric_name, dims):
        if manifold_name in cls.geoopt_manifolds:
            return cls.geoopt_manifolds[manifold_name](dims)

        manifold = cls.sympa_manifolds[manifold_name]
        metric = MetricType.from_str(metric_name)
        return manifold(dims=dims, metric=metric)
