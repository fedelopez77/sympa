"""File with amazon dataset specific functions to collect user-item interactions"""
import json
import gzip


def load_interactions_file(filepath):
    """
    :param filepath: file to 5-core amazon review file
    :return: dict of uid: interactions, sorted by ascending date
    """
    samples = {}
    with gzip.open(str(filepath), 'r') as f:
        for review in map(json.loads, f):
            uid = review["reviewerID"]
            iid = review["asin"]
            timestamp = review["unixReviewTime"]
            if uid in samples:
                samples[uid].append((iid, timestamp))
            else:
                samples[uid] = [(iid, timestamp)]
    sorted_samples = {}
    for uid, ints in samples.items():
        # since a user can interact with the same items several time, the pair (user, item_id)
        # can appear both in train and test and we want to avoid this. Therefore we delete
        # repetitions and keep only the first interaction
        sorted_ints = sorted(ints, key=lambda p: p[1])
        unique_iid_ints = set()
        filtered_ints = []
        for iid, _ in sorted_ints:
            if iid not in unique_iid_ints:
                unique_iid_ints.add(iid)
                filtered_ints.append(iid)
        sorted_samples[uid] = filtered_ints
    return sorted_samples


def load_interactions(reviews_file):
    """
    Loads the interaction file extracted from users' reviews

    :param reviews_file: path to amazon 5-core files
    :return: dict of uid: interactions, sorted by ascending date
    """
    return load_interactions_file(reviews_file)


def build_itemid2name(metadata_file):
    """
    Loads item titles and creates a dictionary

    :param metadata_file: path to amazon product metadata file
    :return: dict of iid: item_title
    """
    with gzip.open(str(metadata_file), 'r') as f:
        return {meta["asin"]: meta.get("title", "None")[:100] for meta in map(json.loads, f)}


def load_reviews(filepath, revs_to_keep=10, separate_title=False, id_key="asin"):
    """
    :param filepath: file to 5-core amazon review file
    :param revs_to_keep: sorts reviews by length and only keeps revs_to_keen longest ones
    :param separate_title: if True, it returns dict with [(title1, review1), (titleN, reviewN)]
    :param id_key: it should be "asin" for item_reviews or "reviewerID" for user_reviews
    :return: dict of iid: list of reviews
    """
    reviews = {}
    with gzip.open(filepath, 'r') as f:
        for line in map(json.loads, f):
            iid = line[id_key]      # id_key should be "asin" or "reviewerID"
            if separate_title:
                this_rev = line.get("reviewText")
            else:
                this_rev = [line.get("summary"), line.get("reviewText")]
                this_rev = ". ".join([x for x in this_rev if x])

            if iid in reviews:
                if separate_title:
                    reviews[iid].append((line.get("summary"), this_rev))
                else:
                    reviews[iid].append(this_rev)
            else:
                if separate_title:
                    reviews[iid] = [(line.get("summary"), this_rev)]
                else:
                    reviews[iid] = [this_rev]

    if separate_title:
        return reviews

    # sorts reviews by length to filter out short ones
    for iid in reviews:
        this_revs = sorted(reviews[iid], key=lambda r: len(r), reverse=True)
        reviews[iid] = this_revs[:revs_to_keep]
    return reviews


def load_metadata_as_text(filepath):
    """
    :param filepath: file to amazon item metadata file
    :return: dict of iid: metadata as one string
    """
    metadata = {}
    with gzip.open(filepath, 'r') as f:
        for line in map(json.loads, f):
            iid = line["asin"]
            this_meta = [line.get("title", "")]
            this_meta += line.get("description", [])
            this_meta += line.get("feature", [])
            if "category" in line:
                cats = line.get("category")
                main_cat = line.get("main_cat")
                if main_cat:
                    try:
                        cats.remove(main_cat)   # main cat is the same for all items, so we remove it
                    except ValueError:
                        pass
                this_meta += cats
            metadata[iid] = ". ".join(this_meta)
    return metadata


def build_text_from_items(dataset_path, reviews_file, metadata_file, use_metadata=True, use_review=True,
                          revs_to_keep=10):
    """
    Build the text that represents each item.
    The text is made of:
     - Item title, and everything that is available from the metadata, which can be (not for all items):
        - features of the item
        - description
        - categories
     - Item reviews created by users.

    :param dataset_path: path to amazon dataset
    :return: dict of iid: list of text for each item
    """
    reviews_file = dataset_path / reviews_file
    print(f"Loading amazon reviews from {reviews_file}")
    texts = load_reviews(reviews_file, revs_to_keep=revs_to_keep)

    metadata_file = dataset_path / metadata_file
    print(f"Loading amazon metadata from {metadata_file}")
    metadata = load_metadata_as_text(metadata_file)

    # We are only interested in the iids in texts, metadata is much larger
    no_meta = 0
    for iid in texts:
        if iid in metadata:
            res = []
            if use_metadata:
                res = [metadata[iid]]
            if use_review:
                res += texts[iid]
            texts[iid] = res
        else:
            no_meta += 1

    print(f"Items with no metadata {no_meta}: {no_meta * 100 / len(texts):.2f}%")
    return texts


def build_polarity_text(dataset_path, reviews_file, item=True):
    """
    Build the text that represents each item.
    The text is made of:
     - Item title, and everything that is available from the metadata, which can be (not for all items):
        - features of the item
        - description
        - categories
     - Item reviews created by users.

    :param dataset_path: path to amazon dataset
    :param if item=True, loads item reviews, if not loads user reviews
    :return: dict of iid: list of text for each item
    """
    from tqdm import tqdm
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    reviews_file = dataset_path / reviews_file
    print(f"Loading amazon reviews from {reviews_file}")
    texts = load_reviews(reviews_file, separate_title=True, id_key="asin" if item else "reviewerID")

    sid = SentimentIntensityAnalyzer()
    polarity_reviews = {}
    for iid in tqdm(texts, total=len(texts), desc="build_polarity_review"):
        res = []
        for title, review in texts[iid]:
            if not title or not review:
                continue
            polarity = sid.polarity_scores(title)
            if polarity["neu"] < 0.5:
                res.append(title + " " + review)

        if not res:
            if not title:
                title = ""
            if not review:
                review = ""
            if not title and not review:
                title = "foo"
            res = [title + " " + review]

        polarity_reviews[iid] = res

    return polarity_reviews


def build_text_from_users(dataset_path, reviews_file, revs_to_keep=10):
    """
    Build the text that represents each user.
    The text is made of all the reviews each user gave

    :param dataset_path: path to amazon dataset
    :return: dict of iid: list of text for each user
    """
    reviews_file = dataset_path / reviews_file
    print(f"Loading amazon reviews from {reviews_file}")
    texts = load_reviews(reviews_file, revs_to_keep=revs_to_keep, id_key="reviewerID")
    return texts


class AmazonItem:
    def __init__(self, metadata):
        """
        :param metadata: dict extracted from json file with amazon item metadata
        """
        self.id = metadata["asin"]
        self.cobuys = metadata.get("also_buy", [])
        self.coviews = metadata.get("also_view", [])
        self.categories = metadata.get("category", [])
        self.brand = metadata.get("brand", "")


def load_metadata(metadata_file):
    """
    Loads metadata as a dict

    :param metadata_file: path to amazon products metadata file
    :return: dict of dicts with metadata
    """
    print(f"Loading amazon metadata from {metadata_file}")
    with gzip.open(str(metadata_file), 'r') as f:
        return [AmazonItem(line) for line in map(json.loads, f)]
