import random
import torch
import numpy as np
import logging
import sys
from datetime import datetime
import pandas as pd
from pathlib import Path
from sympa.config import EPS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_logging(level=logging.DEBUG):
    log = logging.getLogger(__name__)
    log.parent.disabled = True
    log.propagate = False
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
    return torch.all((a - b).abs() < EPS[torch.float32] * factor)


def write_results_to_file(results_file: str, results_data: dict):
    results_data["timestamp"] = [datetime.now().strftime("%Y%m%d%H%M%S")]
    file = Path(results_file)
    pd.DataFrame.from_dict(results_data).to_csv(file, mode="a", header=not file.exists())


def is_prime(a):
    return all(a % i for i in range(2, a))


def scale_triplets(triplets):
    """
    Applies max scaling. Idea taken from Cruceru20.
    In this way we use the max-normalized squared distances.
    This makes all distances to fall between (0, 1]

    :param triplets: iterable with tuples of (src, dst, distance)
    """
    triplets = [(src, dst, dist**2) for src, dst, dist in triplets]
    max_dist = max([dist for _, _, dist in triplets])
    triplets = [(src, dst, dist / max_dist) for src, dst, dist in triplets]
    return triplets


def subsample_triplets(triplets, subsample):
    """
    Keeps the proportion established by 'subsample' of the training triplets.
    Keeps the triplets with the shortest distances, since they represent the local neighborhood of each node.

    :param triplets: iterable with tuples of (src, dst, distance)
    :param subsample: proportion of triplets to subsample. Float number in the range of (0, 1)
    :return: subset of 'triplets'
    """
    distances = np.array([distance for _, _, distance in triplets])
    indexes = np.argpartition(distances, 4)
    sub_length = round(len(triplets) * subsample)
    sub_indexes = indexes[:sub_length]

    sub_triplets = []
    for i in sub_indexes:
        sub_triplets.append(triplets[i])
    return sub_triplets
