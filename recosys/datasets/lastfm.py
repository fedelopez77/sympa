
import random


def load_interactions(dataset_path):
    """
    Maps raw dataset file to a dict of user_id: [item_id, ...] interactions

    :param dataset_path: Path to folder containing interaction file.
        Expected file is 'user_artists.dat' and expected format is:
        userID	artistID	weight      -> header present
        2	51	13883
        2	52	11690

    :return: Dictionary containing users as keys, and a list of items the user interacted with.
    The list is shuffled according to the same random seed every time, in order to choose
    the dev and test items
    """
    filename = "user_artists.dat"
    samples = {}
    with open(dataset_path / filename, 'r') as f:
        next(f)     # ignores header
        for line in f:
            line = line.strip('\n').split()
            uid = line[0]
            iid = line[1]
            if uid in samples:
                samples[uid].append(iid)
            else:
                samples[uid] = [iid]
    shuffled_samples = {}
    SEED = 42
    random.seed(SEED)
    for uid in samples:
        items = samples[uid]
        random.shuffle(items)
        shuffled_samples[uid] = items
    return shuffled_samples
