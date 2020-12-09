
from enum import Enum


class Metric(Enum):
    """Allowed types of metrics that symmetric manifold support"""
    RIEMANNIAN = 0
    FINSLER_ONE = 1
    FINSLER_INFINITY = 2
