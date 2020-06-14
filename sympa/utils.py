import random
import torch
import numpy as np
import logging
import sys
from sympa.config import EPS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


def row_sort(x, indexes):
    """
    Sorts a 2D tensor according to indices, where indices can have a different order for each row
    :param X: b x n: a 2D input tensor
    :param indexes: b x n: a 2D tensor with indices in each row to reorder X
    :return: X sorted, where the row i of X is sorted according to indexes[i].
    """
    d1, d2 = x.size()
    ret = x[
        torch.arange(d1).unsqueeze(1).repeat((1, d2)).flatten(),
        indexes.flatten()
    ].view(d1, d2)
    return ret


def assert_all_close(a, b, factor=1):
    return torch.all((a - b).abs() < EPS[a.dtype] * factor)
