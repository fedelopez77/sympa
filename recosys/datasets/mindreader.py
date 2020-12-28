
import random


def load_interactions(dataset_path):
    """
    Maps raw dataset file to a dict of user_id: [item_id, ...] interactions

    :param dataset_path: Path to folder containing interaction file.
        Expected file is 'ratings.csv' and expected format is:
        ,userId,uri,isItem,sentiment        -> header present
        0,d50f84c0-17cd-11ea-bd9f-33d41a12d743,http://www.wikidata.org/entity/Q191104,False,1
        1,d50f84c0-17cd-11ea-bd9f-33d41a12d743,http://www.wikidata.org/entity/Q44578,True,1

    :return: Dictionary containing users as keys, and a list of items the user interacted with.
    The list is shuffled according to the same random seed every time, in order to choose
    the dev and test items
    """
    filename = "ratings.csv"
    samples = {}
    with open(dataset_path / filename, 'r') as f:
        next(f)     # ignores header
        for line in f:
            line = line.strip('\n').split(",")
            is_item = line[3]
            sentiment = line[4]
            if is_item == "False" or sentiment == "-1": continue    # does not consider "disliked" items
            uid = line[1]
            iid = line[2]
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
