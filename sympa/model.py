import torch
import torch.nn as nn
from sympa.utils import get_logging
from sympa import config
import geoopt as gt
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold
from sympa.manifolds import symmetric_math as smath


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
            self.embeds = smath.sym_make_symmetric(self.embeds)


class EuclideanEmbeddings(Embeddings):
    def __init__(self, num_embeddings, embedding_dim, manifold=None):
        _manifold = manifold if manifold is not None else gt.manifolds.Euclidean(ndim=1)
        _embeds = torch.Tensor(num_embeddings, embedding_dim)
        super().__init__(num_embeddings, embedding_dim, _manifold, _embeds)


class Model(nn.Module):
    def __init__(self, args, graph_distances):
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

        self.graph_distances = graph_distances

    def forward(self, input_index):
        """
        :param input_index: tensor with indexes to process
        :return: loss
        """
        src, dst = self.get_src_and_dst_from_seq(input_index)
        manifold_distances = self.distance(src, dst)
        graph_distances = self.graph_distances[src, dst]

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

    def distortion(self, input_index):
        """
        Distortion is computed as:
            distortion(u,v) = |d_s(u,v) - d_g(u,v)| / d_g(u,v)

        :param input_index: tensor with indexes to process
        :return: distortion
        """
        src, dst = self.get_src_and_dst_from_seq(input_index)
        manifold_dists = self.distance(src, dst)
        graph_dists = self.graph_distances[src, dst]

        distortion = torch.abs(manifold_dists - graph_dists) / graph_dists

        return distortion

    def get_src_and_dst_from_seq(self, input_index):
        """
        :param input_index: tensor with batch of indexes of points: shape: b
        :return: two tensors of len b * (n - 1) with the pairs src[i], dst[i] at each element i
        For each point in input_index, it should compute the distance with all other nodes
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
