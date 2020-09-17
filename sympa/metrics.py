
import torch


class AverageDistortionMetric:

    def __init__(self):
        pass

    def calculate_metric(self, graph_distances, manifold_distances):
        """
        Distortion is computed as:
            distortion(u,v) = |d_s(u,v) - d_g(u,v)| / d_g(u,v)

        :param graph_distances: distances in the graph (b, 1)
        :param manifold_distances: distances in the manifold (b, 1)
        :return: average_distortion: tensor of (1)
        """
        distortion = torch.abs(manifold_distances - graph_distances) / graph_distances
        return distortion
