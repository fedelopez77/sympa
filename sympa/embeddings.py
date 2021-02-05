import abc
import torch
import torch.nn as nn
from sympa import config
import geoopt as gt
from sympa.manifolds.metric import Metric
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold


class Embeddings(nn.Module, abc.ABC):
    """Abstract Embedding layer that operates with embeddings as a Manifold parameter"""

    def __init__(self, num_embeddings, embedding_dim, manifold, _embeds):
        """
        :param num_embeddings: number of elements in the table
        :param embedding_dim: dimensionality of embeddings.
        If it is vector embeddings, it is the amount of dimensions in the vector.
        If it is Matrix embeddings, it is the last dimension on the matrix
        :param manifold: Instance of Geoopt Manifold
        :param _embeds: tensor with initial value for embeddings
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
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


class ComplexSymmetricMatrixEmbeddings(Embeddings):

    def __init__(self, num_embeddings, embedding_dim, manifold):
        """
        Represents Embeddings of Complex Symemtric Matrices.
        The shape of the embedding table is: num_embeddings x 2 x embedding_dim x embedding_dim
        The second dimension is 2, since the element 0 represents the real part,
        and the element 1 represents the imaginary part of the matrices.

        :param num_embeddings: number of elements in the table
        :param embedding_dim: dimensionality of matrix embeddings.
        :param manifold: UpperHalfManifold or BoundedDomainManifold
        """
        _embeds = manifold.random(num_embeddings, from_=-config.INIT_EPS, to=config.INIT_EPS)
        super().__init__(num_embeddings, embedding_dim, manifold, _embeds)

    def norm(self):
        points = self.embeds.data
        points = points.reshape(len(points), -1)
        return points.norm(dim=-1)


class VectorEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold):
        init_eps = config.INIT_EPS
        _embeds = torch.Tensor(num_embeddings, embedding_dim).uniform_(-init_eps, init_eps)
        super().__init__(num_embeddings, embedding_dim, manifold, _embeds)
        self.proj_embeds()

    def norm(self):
        return self.embeds.data.norm(dim=-1)


########################################## Builder logic ####################################################

def get_euclidean_manifold(dims): return gt.Euclidean(1)
def get_poincare_manifold(dims): return gt.PoincareBall()
def get_lorentz_manifold(dims): return gt.Lorentz()
def get_sphere_manifold(dims): return gt.Sphere()
def get_upper_manifold(dims): return UpperHalfManifold(dims, metric=Metric.RIEMANNIAN.value)
def get_upper_fone_manifold(dims): return UpperHalfManifold(dims, metric=Metric.FINSLER_ONE.value)
def get_upper_finf_manifold(dims): return UpperHalfManifold(dims, metric=Metric.FINSLER_INFINITY.value)
def get_upper_fmin_manifold(dims): return UpperHalfManifold(dims, metric=Metric.FINSLER_MINIMUM.value)
def get_upper_wsum_manifold(dims): return UpperHalfManifold(dims, metric=Metric.WEIGHTED_SUM.value)
def get_bounded_manifold(dims): return BoundedDomainManifold(dims, metric=Metric.RIEMANNIAN.value)
def get_bounded_fone_manifold(dims): return BoundedDomainManifold(dims, metric=Metric.FINSLER_ONE.value)
def get_bounded_finf_manifold(dims): return BoundedDomainManifold(dims, metric=Metric.FINSLER_INFINITY.value)
def get_bounded_fmin_manifold(dims): return BoundedDomainManifold(dims, metric=Metric.FINSLER_MINIMUM.value)
def get_bounded_wsum_manifold(dims): return BoundedDomainManifold(dims, metric=Metric.WEIGHTED_SUM.value)


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


class ManifoldBuilder:

    _manifolds = {
        "euclidean": get_euclidean_manifold,
        "poincare": get_poincare_manifold,
        "lorentz": get_lorentz_manifold,
        "sphere": get_sphere_manifold,
        "prod-hysph": get_prod_hysph_manifold,
        "prod-hyhy": get_prod_hyhy_manifold,
        "prod-hyeu": get_prod_hyeu_manifold,
        "upper": get_upper_manifold,
        "upper-fone": get_upper_fone_manifold,
        "upper-finf": get_upper_finf_manifold,
        "upper-fmin": get_upper_fmin_manifold,
        "upper-wsum": get_upper_wsum_manifold,
        "bounded": get_bounded_manifold,
        "bounded-fone": get_bounded_fone_manifold,
        "bounded-finf": get_bounded_finf_manifold,
        "bounded-fmin": get_bounded_fmin_manifold,
        "bounded-wsum": get_bounded_wsum_manifold
    }

    @classmethod
    def get_manifold(cls, name, dims):
        return cls._manifolds[name](dims)


class EmbeddingsBuilder:
    @classmethod
    def get_embeddings(cls, name: str, num_points: int, dims: int, manifold):
        embed_table = cls._get_table(name)
        return embed_table(num_embeddings=num_points, embedding_dim=dims, manifold=manifold)

    @classmethod
    def _get_table(cls, model_name: str):
        if "upper" in model_name or "bounded" in model_name:
            return ComplexSymmetricMatrixEmbeddings
        if model_name in {"euclidean", "poincare", "lorentz", "sphere", "prod-hysph", "prod-hyhy", "prod-hyeu"}:
            return VectorEmbeddings
        raise ValueError(f"Unrecognized embedding model: {model_name}")
