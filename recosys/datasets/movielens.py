"""File with movie lens dataset specific functions"""


def movielens_to_dict(dataset_path):
    """
    Maps raw dataset file to a Dictonary.

    :param dataset_path: Path to folder containing interaction file.
        Expected file and formats are:
        ml-1m: file: 'ratings.dat', format: uid::iid::rate::time
        ml-100k: file: 'u.data', format: uid \t iid \t rate \t time -> tsv

    :return: Dictionary containing users as keys, and a list of items the user
      interacted with, sorted by the time of interaction.
    """
    filename = "ratings.dat"
    sep = "::"
    if "100k" in str(dataset_path):
        filename = "u.data"
        sep = "\t"
    samples = {}
    with open(dataset_path / filename, 'r') as f:
        for line in f:
            line = line.strip('\n').split(sep)
            uid = line[0]
            iid = line[1]
            timestamp = int(line[3])
            if uid in samples:
                samples[uid].append((iid, timestamp))
            else:
                samples[uid] = [(iid, timestamp)]
    sorted_samples = {}
    for uid in samples:
        sorted_items = sorted(samples[uid], key=lambda p: p[1])
        sorted_samples[uid] = [pair[0] for pair in sorted_items]
    return sorted_samples


def build_movieid2title(dataset_path):
    """Builds a mapping between item ids and the title of each item."""
    filename = "movies.dat"
    movieid2title = {}
    with open(dataset_path / filename, "r", encoding="ISO-8859-1") as f:
        for line in f:
            line = line.strip("\n").split("::")
            movieid2title[line[0]] = line[1]
    return movieid2title


def build_texts_from_movies(path_to_movie_dat):
    """
    Extracts genre text from movies.dat to create semantic embeddings

    :param path_to_movie_dat:
    :return: dict of text list keyed by movie_id
    """
    texts = {}
    with open(path_to_movie_dat, "r", encoding="ISO-8859-1") as f:
        for line in f:
            movie_id, title_and_year, genres = line.strip("\n").split("::")
            title = title_and_year[:-7]
            # year = title_and_year[-5:-1]
            sorted_genres = sorted(genres.split("|"))
            texts[movie_id] = [title] + sorted_genres
    return texts


