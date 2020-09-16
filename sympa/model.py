import torch
import torch.nn as nn
from sympa.utils import get_logging
from sympa import config
import geoopt as gt
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold
from sympa.math import symmetric_math as smath
import abc

log = get_logging()


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


class Model(nn.Module):
    """Graph embedding model that operates on different spaces"""
    def __init__(self, args):
        super().__init__()

        self.num_points = args.num_points
        self.all_points = list(range(self.num_points))
        if args.model == "euclidean":
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=gt.Euclidean(1))
        elif args.model == "poincare":
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=gt.PoincareBall())
        elif args.model == "upper":
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims,
                                                               manifold=UpperHalfManifold(args.dims))
        elif args.model == "bounded":
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims,
                                                               manifold=BoundedDomainManifold(args.dims))
        self.manifold = self.embeddings.manifold

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
            all_points = [a, b, c, d]
            input_index = [c, a]
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

    def check_all_points(self):
        return self.embeddings.check_all_points()

    def embeds_norm(self):
        return self.embeddings.norm()
