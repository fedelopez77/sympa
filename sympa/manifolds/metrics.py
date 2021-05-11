from abc import ABC, abstractmethod
from enum import Enum
import torch


class MetricType(Enum):
    """Allowed types of metrics that Siegel manifolds support"""
    RIEMANNIAN = "riem"
    FINSLER_ONE = "fone"
    FINSLER_INFINITY = "finf"
    FINSLER_MINIMUM = "fmin"
    WEIGHTED_SUM = "wsum"

    @staticmethod
    def from_str(label):
        types = {t.value: t for t in list(MetricType)}
        return types[label]


class Metric(ABC):

    def __init__(self, dims: int):
        self.dims = dims

    @abstractmethod
    def compute_metric(self, v: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def get(cls, type: MetricType, dims: int):
        metrics_map = {
            MetricType.RIEMANNIAN: RiemannianMetric,
            MetricType.FINSLER_ONE: FinslerOneMetric,
            MetricType.FINSLER_INFINITY: FinslerInfinityMetric,
            MetricType.FINSLER_MINIMUM: FinslerMinimumEntropyMetric,
            MetricType.WEIGHTED_SUM: FinslerWeightedSumMetric
        }

        return metrics_map[type](dims)


class RiemannianMetric(Metric):

    def compute_metric(self, v: torch.Tensor, keepdim=False) -> torch.Tensor:
        """
        Given v_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        the Riemannian distance is given by the summation of this values
        :param v: b x n: v_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        :return: b x 1: Riemannian distance
        """
        res = torch.norm(v, dim=-1, keepdim=keepdim)
        return res


class FinslerOneMetric(Metric):

    def compute_metric(self, v: torch.Tensor, keepdim=False) -> torch.Tensor:
        """
        Given d_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        the Finsler distance one (F_{1})is given by the summation of this values
        :param v: b x n: d_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        :return: b x 1: Finsler distance
        """
        res = torch.sum(v, dim=-1, keepdim=keepdim)
        return res


class FinslerInfinityMetric(Metric):

    def compute_metric(self, v: torch.Tensor, keepdim=False) -> torch.Tensor:
        """
        The Finsler distance infinity (F_{\infty}) is given by d_i = log((1 + r_i) / (1 - r_i)),
        with r_n the largest eigenvalues of the crossratio matrix,
        :param v: b x n: v_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
                where r_0 the largest eigenvalue and r_{n-1} the smallest
        :return: b x 1: Finsler distance
        """
        res = v[:, -1]
        if keepdim:
            return res.reshape((-1, 1))
        return res


class FinslerMinimumEntropyMetric(Metric):

    def __init__(self, dims: int):
        super().__init__(dims)
        factor = 2
        self.weights = factor * (dims + 1 - torch.arange(start=dims + 1, end=1, step=-1).unsqueeze(0))

    def compute_metric(self, v: torch.Tensor, keepdim=False) -> torch.Tensor:
        """
        Given v_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        the Finsler distance of minimum entropy (F_{min})is given by the summation of the weighted values
        \sum  2 * (n + 1 - i) * d_i
        :param v: b x n: v_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        :return: b x 1: Finsler distance
        """
        res = torch.sum(self.weights * v, dim=-1, keepdim=keepdim)
        return res


class FinslerWeightedSumMetric(Metric, torch.nn.Module):

    def __init__(self, dims):
        torch.nn.Module.__init__(self)
        Metric.__init__(self, dims)
        self.weights = torch.nn.parameter.Parameter(torch.ones((1, dims)))

    def compute_metric(self, v: torch.Tensor, keepdim=False) -> torch.Tensor:
        """
        Given v_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        we learn weights for each eigenvalue and apply a weighted average such that:
        distance = \sum  w_i * d_i
        :param v: b x n: d_i = log((1 + d_i) / (1 - d_i)), with d_i the eigenvalues of the crossratio matrix,
        :return: b x 1: Finsler distance
        """
        weights = torch.nn.functional.relu(self.weights)    # 1 x n
        res = weights * v
        res = torch.sum(res, dim=-1, keepdim=keepdim)
        return res
