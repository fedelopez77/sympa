import torch
import torch.nn as nn
from sympa.embeddings import EmbeddingsBuilder, ManifoldBuilder


class Model(nn.Module):
    """Graph embedding model that operates on different Manifolds"""
    def __init__(self, args):
        super().__init__()
        self.manifold = ManifoldBuilder.get_manifold(name=args.model, dims=args.dims)
        self.embeddings = EmbeddingsBuilder.get_embeddings(args.model, args.num_points, args.dims, self.manifold)
        self.scale_coef = args.scale_coef
        self.scale = torch.nn.Parameter(torch.Tensor([self.scale_coef * args.scale_init]),
                                        requires_grad=args.train_scale == 1)

    def forward(self, input_triplet):
        """
        Calculates and returns the distance in the manifold of the points given as a
        triplet of the form: (src_id, dst_id, graph_distance)
        'graph_distance' is ignored in this case.

        :param input_triplet: tensor with indexes of embeddings to process. (b, 3)
        :return: distances b
        """
        src_index, dst_index = input_triplet[:, 0], input_triplet[:, 1]
        src_embeds = self.embeddings(src_index)                       # b x 2 x n x n or b x n
        dst_embeds = self.embeddings(dst_index)                       # b x 2 x n x n

        distances = self.distance(src_embeds, dst_embeds)
        return distances * self.get_scale()

    def distance(self, src_embeds, dst_embeds):
        """
        :param src_embeds, dst_embeds: embeddings of nodes in the manifold.
        In complex matrix spaces, it will be of the shape b x 2 x n x n. In Vector spaces it will be b x n
        :return: tensor of b with distances from each src to each dst
        """
        return self.manifold.dist(src_embeds, dst_embeds)   # b x 1

    def get_scale(self):
        return (self.scale / self.scale_coef).clamp_min(0.1)     #torch.log(self.scale + 1).clamp_min(0.1)

    def check_all_points(self):
        return self.embeddings.check_all_points()

    def embeds_norm(self):
        return self.embeddings.norm()
