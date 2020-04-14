import random
import torch
import numpy as np
import logging
import sys


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


def get_src_and_dst_from_seq(input_index):
    """
    :param input_index: tensor with batch of indexes of points: shape: b
    :return: two tensors of len b * (b - 1) / 2 with the pairs src[i], dst[i] at each element i
    Example:
        input_index = [a, b, c, d]
        src: [a, a, a, b, b, c]
        dst: [b, c, d, c, d, d]
    """
    src, dst = [], []
    device = input_index.device
    input_index = input_index.tolist()
    for i, id_a in enumerate(input_index):
        for id_b in input_index[i + 1:]:
            src.append(id_a)
            dst.append(id_b)

    src_index = torch.LongTensor(src).to(device)
    dst_index = torch.LongTensor(dst).to(device)
    return src_index, dst_index
