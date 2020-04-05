import torch
import torch.nn as nn
from sympa.utils import get_logging
from sympa import config
import geoopt as gt
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold
from sympa.manifolds import symmetric_math as smath


log = get_logging()


class Embeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, manifold, weight):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold
        self.weight = gt.ManifoldParameter(weight, manifold=self.manifold)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.similarity_layer.weight, a=-config.INIT_EPS, b=config.INIT_EPS)

    def forward(self, input_index):
        return self.weight[input_index]


class ComplexEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold=UpperHalfManifold()):
        _weight = torch.Tensor(num_embeddings, 2, embedding_dim, embedding_dim)
        _weight = smath.sym_make_symmetric(_weight)

        super().__init__(num_embeddings, embedding_dim, manifold, _weight)


class EuclideanEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim):
        _manifold = gt.manifolds.Euclidean(ndim=embedding_dim)
        _weight = torch.Tensor(num_embeddings, embedding_dim)
        super().__init__(num_embeddings, embedding_dim, _manifold, _weight)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embeddings = ComplexEmbeddings()

