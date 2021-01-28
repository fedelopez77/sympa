import torch
import torch.nn as nn
from sympa.utils import get_logging
import geoopt as gt
from sympa.manifolds.metric import Metric
from sympa.manifolds import BoundedDomainManifold, UpperHalfManifold, SymmetricPositiveDefinite
from sympa.embeddings import VectorEmbeddings, ComplexSymmetricMatrixEmbeddings

log = get_logging()


class RecoSys(nn.Module):
    """Recommender system model that operates on different Manifolds"""
    def __init__(self, args):
        super().__init__()
        if args.model == "euclidean":
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=gt.Euclidean(1))
        elif args.model == "poincare":
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=gt.PoincareBall())
        elif args.model == "lorentz":
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=gt.Lorentz())
        elif args.model == "sphere":
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=gt.Sphere())
        elif args.model == "prod-hysph":
            poincare = gt.PoincareBall()
            sphere = gt.Sphere()
            product = gt.ProductManifold((poincare, args.dims // 2), (sphere, args.dims // 2))
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=product)
        elif args.model == "prod-hyhy":
            poincare = gt.PoincareBall()
            product = gt.ProductManifold((poincare, args.dims // 2), (poincare, args.dims // 2))
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=product)
        elif args.model == "prod-hyeu":
            poincare = gt.PoincareBall()
            euclidean = gt.Euclidean(1)
            product = gt.ProductManifold((poincare, args.dims // 2), (euclidean, args.dims // 2))
            self.embeddings = VectorEmbeddings(args.num_points, args.dims, manifold=product)
        elif args.model == "upper":
            manifold = UpperHalfManifold(args.dims, metric=Metric.RIEMANNIAN.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "upper-fone":
            manifold = UpperHalfManifold(args.dims, metric=Metric.FINSLER_ONE.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "upper-finf":
            manifold = UpperHalfManifold(args.dims, metric=Metric.FINSLER_INFINITY.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "upper-fmin":
            manifold = UpperHalfManifold(args.dims, metric=Metric.FINSLER_MINIMUM.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "bounded":
            manifold = BoundedDomainManifold(args.dims, metric=Metric.RIEMANNIAN.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "bounded-fone":
            manifold = BoundedDomainManifold(args.dims, metric=Metric.FINSLER_ONE.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "bounded-finf":
            manifold = BoundedDomainManifold(args.dims, metric=Metric.FINSLER_INFINITY.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "bounded-fmin":
            manifold = BoundedDomainManifold(args.dims, metric=Metric.FINSLER_MINIMUM.value)
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        elif args.model == "spd":
            manifold = SymmetricPositiveDefinite()
            self.embeddings = ComplexSymmetricMatrixEmbeddings(args.num_points, args.dims, manifold=manifold)
        else:
            raise ValueError(f"Unrecognized model option: {args.model}")

        self.manifold = self.embeddings.manifold
        self.bias_lhs = torch.nn.Parameter(torch.zeros(args.num_points), requires_grad=args.train_bias == 1)
        self.bias_rhs = torch.nn.Parameter(torch.zeros(args.num_points), requires_grad=args.train_bias == 1)

    def forward(self, input_triplet):
        """
        Calculates and returns the score for a pair (head, tail), based on the distances in the space.

        :param input_triplet: tensor with indexes of embeddings to process. (b, 2)
        :return: scores: b
        """
        lhs_index, rhs_index = input_triplet[:, 0], input_triplet[:, -1]
        lhs_embeds = self.embeddings(lhs_index)                       # b x 2 x n x n or b x n
        rhs_embeds = self.embeddings(rhs_index)                       # b x 2 x n x n
        lhs_bias = self.bias_lhs[lhs_index]                           # b
        rhs_bias = self.bias_rhs[rhs_index]                           # b

        sq_distances = self.distance(lhs_embeds, rhs_embeds) ** 2
        scores = lhs_bias + rhs_bias - sq_distances

        return scores

    def distance(self, src_embeds, dst_embeds):
        """
        :param src_embeds, dst_embeds: embeddings of nodes in the manifold.
        In complex matrix spaces, it will be of the shape b x 2 x n x n. In Vector spaces it will be b x n
        :return: tensor of b with distances from each src to each dst
        """
        return self.manifold.dist(src_embeds, dst_embeds)   # b x 1

    def check_all_points(self):
        return self.embeddings.check_all_points()

    def embeds_norm(self):
        return self.embeddings.norm()
