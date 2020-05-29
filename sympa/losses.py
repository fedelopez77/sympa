
import torch


class AverageDistortionLoss:

    def __init__(self, graph_distances):
        self.graph_distances = graph_distances

    def calculate_loss(self, src_index, dst_index, src_embeds, dst_embeds, distance_fn):
        """
        :param src_index, dst_index: indexes of embeddings to process
        :param src_embeds, dst_embeds: embeddings of nodes in the manifold.
        In symmetric spaces, it will be of the shape b x 2 x n x n. In Euclidean space it will be b x n
        :param distance_fn: function to calculate the distance in the manifold
        :return: loss
        """
        manifold_distances = distance_fn(src_embeds, dst_embeds)
        graph_distances = self.graph_distances[src_index, dst_index]

        loss = torch.pow(manifold_distances / graph_distances, 2)
        loss = torch.abs(loss - 1)
        loss = loss.sum()

        return loss
