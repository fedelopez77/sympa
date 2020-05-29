
import torch


class AverageDistortionMetric:

    def __init__(self, graph_distances):
        self.graph_distances = graph_distances

    def calculate_metric(self, src_index, dst_index, src_embeds, dst_embeds, distance_fn):
        """
        Distortion is computed as:
            distortion(u,v) = |d_s(u,v) - d_g(u,v)| / d_g(u,v)

        :param src_index, dst_index: indexes of embeddings to process
        :param src_embeds, dst_embeds: embeddings of nodes in the manifold.
        In symmetric spaces, it will be of the shape b x 2 x n x n. In Euclidean space it will be b x n
        :param distance_fn: function to calculate the distance in the manifold
        :return: loss
        """
        manifold_distances = distance_fn(src_embeds, dst_embeds)
        graph_distances = self.graph_distances[src_index, dst_index]

        distortion = torch.abs(manifold_distances - graph_distances) / graph_distances

        return distortion
