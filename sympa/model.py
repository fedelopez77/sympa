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
        self.embeds = gt.ManifoldParameter(weight, manifold=self.manifold)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.similarity_layer.weight, a=-config.INIT_EPS, b=config.INIT_EPS)

    def forward(self, input_index):
        return self.embeds[input_index]


class ComplexEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold=UpperHalfManifold()):
        _embeds = torch.Tensor(num_embeddings, 2, embedding_dim, embedding_dim)
        super().__init__(num_embeddings, embedding_dim, manifold, _embeds)
        with torch.no_grad():
            self.embeds = smath.sym_make_symmetric(self.embeds)


class EuclideanEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold=None):
        _manifold = manifold if manifold is not None else gt.manifolds.Euclidean(ndim=embedding_dim)
        _embeds = torch.Tensor(num_embeddings, embedding_dim)
        super().__init__(num_embeddings, embedding_dim, _manifold, _embeds)


class Model(nn.Module):
    def __init__(self, args, graph):
        """
        :param args:
        :param graph: tensor of n x n with n = args.num_points with all the distances in the graph
        """
        super().__init__()

        if args.model == "euclidean":
            self.embeddings = EuclideanEmbeddings(args.num_points, args.dims)
            self.manifold = self.embeddings.manifold
        elif args.model == "upper":
            self.manifold = UpperHalfManifold()
            self.embeddings = ComplexEmbeddings(args.num_points, args.dims, manifold=self.manifold)
        elif args.model == "bounded":
            self.manifold = BoundedDomainManifold()
            self.embeddings = ComplexEmbeddings(args.num_points, args.dims, manifold=self.manifold)

        self.graph = graph

    def forward(self, input_index):
        """
        :param input_index: tensor with indexes to process
        :return: loss
        """
        src, dst = self.get_src_and_dst_from_seq(input_index)                   # tensors of length e
        manifold_distances = self.distance(src, dst)
        graph_distances = self.graph[src, dst]

        loss = torch.pow(manifold_distances / graph_distances, 2)
        loss = torch.abs(loss - 1)
        loss = loss.sum()

        return loss

    def distance(self, src, dst):
        """
        :param src, dst: tensors of len b with ids of src and dst points to calculate distances
        :return: tensor of b with distances from each src to each dst
        """
        src_embeds = self.embeddings(src)                   # b x 2 x n x n
        dst_embeds = self.embeddings(dst)                   # b x 2 x n x n

        return self.manifold.dist(src_embeds, dst_embeds)   # b x 1

    def average_distortion(self, input_index):
        """
        Distortion is computed as:
            distortion(u,v) = |d_s(u,v) - d_g(u,v)| / d_g(u,v)

        :param input_index: tensor with indexes to process
        :return: average distortion
        """
        src, dst = self.get_src_and_dst_from_seq(input_index)  # tensors of length e
        manifold_distances = self.distance(src, dst)
        graph_distances = self.graph[src, dst]

        distortion = torch.abs(manifold_distances - graph_distances) / graph_distances
        avg_distortion = distortion.mean()

        return avg_distortion
