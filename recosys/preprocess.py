import os
import sys
sys.path.append(os.path.abspath('../sympa'))
sys.path.append(os.path.abspath('.'))
import pickle
import numpy as np
import random
from pathlib import Path
import argparse
from sympa.utils import set_seed
from recosys.datasets import amazon, movielens, lastfm, mindreader, meetup


def plot_graph(samples):
    """Plot user-item graph, setting different colors for items and users."""
    import networkx as nx
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    for uid, ints in samples.items():
        for iid in ints:
            graph.add_edge(uid, iid)

    color_map = ["red" if node in samples else "blue" for node in graph]
    fig = plt.figure()
    pos = nx.spring_layout(graph, iterations=100)
    nx.draw(graph, pos, ax=fig.add_subplot(111), node_size=20, node_color=color_map)
    plt.show()


def map_raw_ids_to_sequential_ids(samples):
    """
    For each unique user or item id, this function creates a mapping to a sequence of number starting in 0.
    This will be the index of the embeddings in the model.

    Items ids will be from 0 to n_items - 1.
    Users ids will be from n_items to n_items + n_users - 1
    This condition is required to later build the distance matrix

    :param samples: dict of <user_id1>: [<item_id1>, <item_id2>, ...]
    :return: dicts of {<user_idX>: indexY} and {<item_idX>: indexW}
    """
    uid2id, iid2id = {}, {}
    sorted_samples = sorted(samples.items(), key=lambda x: x[0])
    # first sets items ids only
    for _, ints in sorted_samples:
        sorted_ints = sorted(ints)
        for iid in sorted_ints:
            if iid not in iid2id:
                iid2id[iid] = len(iid2id)
    # users ids come after item ids
    for uid, _ in sorted_samples:
        if uid not in uid2id:
            uid2id[uid] = len(uid2id) + len(iid2id)

    return uid2id, iid2id


def create_splits(samples, do_random=False, seed=42):
    """
    Splits (user, item) dataset to train, dev and test.

    :param samples: Dict of sorted examples.
    :param do_random: Bool whether to extract dev and test by random sampling. If False, dev, test are the last two
        items per user.
    :return: examples: Dictionary with 'train','dev','test' splits as numpy arrays
        containing corresponding (user_id, item_id) pairs
    """
    train, dev, test = [], [], []
    for uid, ints in samples.items():
        if do_random:
            random.seed(seed)
            random.shuffle(ints)
        if len(ints) >= 3:
            test.append((uid, ints[-1]))
            dev.append((uid, ints[-2]))
            for iid in ints[:-2]:
                train.append((uid, iid))
        else:
            for iid in ints:
                train.append((uid, iid))
    return {
        'samples': samples,
        'train': np.array(train).astype('int64'),
        'dev': np.array(dev).astype('int64'),
        'test': np.array(test).astype('int64')
    }


def save_as_pickle(save_path, data):
    with open(str(save_path), 'wb') as fp:
        pickle.dump(data, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="recosys-preprocess.py")
    parser.add_argument('--run_id', default='software', type=str, help='Name of prep to store')
    parser.add_argument('--item', default='meetup', type=str, help='Item to process')
    parser.add_argument('--dataset_path', default='data/recosys/meetup/CA', type=str, help='Path to raw dataset')
    parser.add_argument('--amazon_reviews', default='Software_5.json.gz', type=str,
                        help='Name of the 5-core amazon reviews file')
    parser.add_argument('--plot_graph', default=0, type=int, help='Plots the user-item graph')
    parser.add_argument('--shuffle', default=0, type=int, help='Whether to shuffle the interactions or not')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_path = Path(args.dataset_path)

    if args.item == "amazon":
        samples = amazon.load_interactions(dataset_path / args.amazon_reviews)
    elif args.item == "movie":
        samples = movielens.movielens_to_dict(dataset_path)
    elif args.item == "lastfm":
        samples = lastfm.load_interactions(dataset_path)
    elif args.item == "mindreader":
        samples = mindreader.load_interactions(dataset_path)
    elif args.item == "meetup":
        samples = meetup.load_interactions(dataset_path)
    else:
        raise ValueError(f"Unknown item: {args.item}")

    if args.plot_graph:
        plot_graph(samples)
        exit(0)

    uid2id, iid2id = map_raw_ids_to_sequential_ids(samples)

    id_samples = {}
    for uid, ints in samples.items():
        id_samples[uid2id[uid]] = [iid2id[iid] for iid in ints]

    data = create_splits(id_samples, do_random=args.shuffle, seed=args.seed)
    data["id2uid"] = {v: k for k, v in uid2id.items()}
    data["id2iid"] = {v: k for k, v in iid2id.items()}
    total = len(data['train']) + len(data['dev']) + len(data['test'])
    print(f"Users: {len(uid2id)}, items: {len(iid2id)}, total interactions: {total}")
    print(f"Density: {total / (len(uid2id) * len(iid2id)) * 100:.2f}%")
    print(f"Process dataset. Train: {len(data['train'])}, dev: {len(data['dev'])}, test: {len(data['test'])}")

    prep_path = Path("data/recosys/prep")
    prep_path.mkdir(parents=True, exist_ok=True)
    to_save_dir = prep_path / args.run_id
    to_save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving in {to_save_dir}")
    save_as_pickle(to_save_dir / f'{args.run_id}.pickle', data)
    print("Done!")
