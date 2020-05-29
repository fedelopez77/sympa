import torch
import torch.nn as nn
from sympa.utils import get_logging
from sympa import config
import geoopt as gt
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold
from sympa.math import symmetric_math as smath

log = get_logging()


class Embeddings(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, manifold, _embeds):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.manifold = manifold
        self.embeds = gt.ManifoldParameter(_embeds, manifold=self.manifold)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.embeds.data, a=-config.INIT_EPS, b=config.INIT_EPS)

    def forward(self, input_index):
        return self.embeds[input_index]


class ComplexEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold=UpperHalfManifold()):
        _embeds = torch.Tensor(num_embeddings, 2, embedding_dim, embedding_dim)
        super().__init__(num_embeddings, embedding_dim, manifold, _embeds)
        with torch.no_grad():
            self.embeds = smath.to_symmetric(self.embeds)
            self.embeds = self.manifold.projx(self.embeds)


class EuclideanEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold=None):
        _manifold = manifold if manifold is not None else gt.manifolds.Euclidean(ndim=1)
        _embeds = torch.Tensor(num_embeddings, embedding_dim)
        super().__init__(num_embeddings, embedding_dim, _manifold, _embeds)


class Model(nn.Module):
    def __init__(self, args):
        """
        :param args:
        :param graph: tensor of n x n with n = args.num_points with all the distances in the graph
        """
        super().__init__()

        self.num_points = args.num_points
        self.all_points = list(range(self.num_points))
        if args.model == "euclidean":
            self.embeddings = EuclideanEmbeddings(args.num_points, args.dims)
            self.manifold = self.embeddings.manifold
        elif args.model == "upper":
            self.manifold = UpperHalfManifold()
            self.embeddings = ComplexEmbeddings(args.num_points, args.dims, manifold=self.manifold)
        elif args.model == "bounded":
            self.manifold = BoundedDomainManifold()
            self.embeddings = ComplexEmbeddings(args.num_points, args.dims, manifold=self.manifold)

    def forward(self, input_index):
        """
        :param input_index: tensor with indexes to process: b
        :return: src and dst indexes. Src and dst embeddings
        """
        src_index, dst_index = self.get_src_and_dst_from_seq(input_index)
        src_embeds = self.embeddings(src_index)                       # b x 2 x n x n or b x n
        dst_embeds = self.embeddings(dst_index)                       # b x 2 x n x n

        return src_index, dst_index, src_embeds, dst_embeds

    def distance(self, src_embeds, dst_embeds):
        """
        :param src_embeds, dst_embeds: embeddings of nodes in the manifold.
        In symmetric spaces, it will be of the shape b x 2 x n x n. In Euclidean space it will be b x n
        :return: tensor of b with distances from each src to each dst
        """
        return self.manifold.dist(src_embeds, dst_embeds)   # b x 1

    def get_src_and_dst_from_seq(self, input_index):
        """
        :param input_index: tensor with batch of indexes of points: shape: b
        :return: two tensors of len b * (n - 1) with the pairs src[i], dst[i] at each element i

        For each point in input_index, it should compute the distance with all other nodes "greater" than the point.
        This is implemented according to the loss equation:
            1 <= i < j <= n
        'input_index' is assumed to be i and this function returns the pairs (i, j) for all j > i

        Example:
            input_index = [c, a]
            all_points = [a, b, c, d]
            src: [c, a, a, a]
            dst: [d, b, c, d]

            input_index = [b, d]
            src: [b, b]
            dst: [c, d]
        """
        src, dst = [], []
        for id_a in input_index.tolist():
            for id_b in self.all_points[id_a + 1:]:
                src.append(id_a)
                dst.append(id_b)

        src_index = torch.LongTensor(src).to(input_index.device)
        dst_index = torch.LongTensor(dst).to(input_index.device)
        return src_index, dst_index
