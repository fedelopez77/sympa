
import torch
import numpy as np
import random
from sympa.utils import set_seed


class RankingBuilder:

    def __init__(self, ini_index, end_index, samples):
        """
        Rank items among a random sample. Ini and end indexes mark the range of ids from where
        the random items can be sampled.
        PRE: ini and end indexes should points only to item ids, NOT users.
        :param ini_index: int
        :param end_index: int
        :param samples: dict of user_id: list of item ids the user has interacted with
        """
        self.ini_index = ini_index
        self.end_index = end_index
        self.samples = samples
        self.seed = 42
        self.candidate_ids = set(range(self.ini_index, self.end_index + 1))

    def rank(self, model, eval_batch, n_random_items=100):
        """
        Ranks eval_item_id over n_random_items items and retrieves its position.

        :param model:
        :param eval_batch: b x 2, with (user_id, eval_item_id).
        :param n_random_items: number of random items to sample
        :return: ranks_random: Numpy array of shape (n_examples, ) containing the rank of each
                example in random setting (ranking against randomly selected random_items items).
                Best ranking: 1 (ranking is 1-numerated)
        """
        inputs = []
        for user_id, _ in eval_batch:
            interacted = set(self.samples[user_id.item()])
            candidates = list(self.candidate_ids - interacted)
            np.random.seed(self.seed)
            random_ids = np.random.choice(candidates, n_random_items, replace=False)

            input_tensor = self.build_input_tensor(user_id, random_ids)     # n x 2
            inputs.append(input_tensor)

        input_tensor = torch.cat(inputs, dim=0)     # b * n x 2
        scores = model(input_tensor)                # b * n
        random_scores = scores.reshape(-1, n_random_items).numpy()

        target_scores = torch.unsqueeze(model(eval_batch), 1).numpy()
        ranking = np.sum((random_scores >= target_scores), axis=1) + 1
        return ranking

    def build_input_tensor(self, user_id, random_ids):
        """
        :param user_id: tensor with user_id
        :param random_ids: list with n random_ids
        :return: tensor of random_ids x 2 with (user_id, random_ids)
        """
        user_ids = torch.unsqueeze(user_id.repeat(len(random_ids)), 1)  # n x 1
        random_ids = torch.unsqueeze(torch.LongTensor(random_ids), 1)   # n x 1
        return torch.cat((user_ids, random_ids), dim=-1)


def rank_to_metric(rank, at_k=10):
    """
    Computes metrics and returns them as dict
    :param rank: Numpy array of shape (n_examples, ) containing the rank of each example
    :return: hr at_k, ndcg at_k, and mean reciprocal ranking
    """
    mean_reciprocal_rank = np.mean(1. / rank)

    less_or_equal_k = rank <= at_k
    hr_at_k = np.mean(less_or_equal_k) * 100

    ndcg_at_k = 1 / np.log2(rank + 1)
    ndcg_at_k = np.where(less_or_equal_k, ndcg_at_k, 0)
    ndcg_at_k = np.mean(ndcg_at_k)

    return hr_at_k, ndcg_at_k, mean_reciprocal_rank
