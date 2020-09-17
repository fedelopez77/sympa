
import torch


class AverageDistortionLoss:

    def __init__(self):
        pass

    def calculate_loss(self, graph_distances, manifold_distances):
        """
        :param graph_distances: distances in the graph (b, 1)
        :param manifold_distances: distances in the manifold (b, 1)
        :return: loss
        """
        loss = torch.pow(manifold_distances / graph_distances, 2)
        loss = torch.abs(loss - 1)
        loss = loss.sum()
        return loss
