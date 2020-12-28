
import random


def load_interactions(dataset_path):
    """
    Maps raw dataset file to a dict of user_id: [item_id, ...] interactions

    :param dataset_path: Path to folder containing interaction file.
        Expected file is 'train/event_users.txt' + 'test/event_users.txt' and expected format is:
        E_4782 U_4684 U_12766 U_27983 U_31210 U_3836
        E_4780 U_4709 U_32820 U_32821 U_3631 U_12606 U_15839 U_8644 U_8376
        Event_id User_id User_id User_id ...

    :return: Dictionary containing users as keys, and a list of items the user interacted with.
    The list is shuffled according to the same random seed every time, in order to choose
    the dev and test items
    """
    filenames = ["train/event_users.txt", "test/event_users.txt"]
    samples = {}
    for filename in filenames:
        with open(dataset_path / filename, 'r') as f:
            for line in f:
                line = line.strip().split()
                iid = line[0]
                for uid in line[1:]:
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
