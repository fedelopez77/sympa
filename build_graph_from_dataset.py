
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import networkx as nx
import torch


CONFIG = {  # first index, file sep
    "iris": (0, ","),
    "zoo": (1, ","),
    "glass": (1, ","),
    "segment": (0, " "),
    "energy": (0, ",")
}


def load_data(dataset_name):
    path = f"data/{dataset_name}/{dataset_name}.data"
    first_index, sep = CONFIG[dataset_name]
    data = []
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip().split(sep)
            if len(line) > 1:
                if dataset_name == "energy":
                    data.append(list(map(float, line[first_index:-2])))
                    labels.append(round(float(line[-1])))
                else:
                    data.append(list(map(float, line[first_index:-1])))
                    labels.append(line[-1])
    return np.array(data, dtype=np.float32), labels


def standardize_data(data):
    """
    Normalize (standardize) features so that each attribute has mean zero and standard deviation one,
    to then compute cosine distance.

    :param data: list of datapoints
    :return: new_data: numpy array with normalized data
    """
    return StandardScaler().fit_transform(data)


def convert_labels(labels):
    """Given a list of categorical labels, it transforms them into a [0..n_class - 1] format.
    Returns labels in the exact same order that they were provided"""
    return LabelEncoder().fit_transform(labels)


def build_distance_matrix(data):
    """
    Builds distance matrix from features, based on cosine distance
    :param data: numpy array of n_points x n_features data points
    :return: numpy array of n_points x n_points with cosine distance between each datapoint
    """
    norm = np.linalg.norm(data, axis=-1, keepdims=True)
    data = data / norm
    sim_matrix = data @ np.transpose(data)
    dist_matrix = np.clip(1 - sim_matrix, a_min=1e-9, a_max=None)
    return dist_matrix


def build_graph(distance_matrix):
    """Given a symmetric distance matrix, it builds a weighted graph"""
    graph = nx.Graph()
    nodes = len(distance_matrix)
    for i in range(nodes):
        for j in range(i + 1, nodes):
            dist = distance_matrix[i][j]
            graph.add_edge(i, j, weight=f"{dist:.9f}")
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser("build_graph_from_dataset.py")
    parser.add_argument("--data", required=True, type=str, help="Name of prep folder")
    args = parser.parse_args()
    data, labels = load_data(args.data)
    data = standardize_data(data)
    labels = convert_labels(labels)

    distance_matrix = build_distance_matrix(data)
    graph = build_graph(distance_matrix)

    path = f"data/{args.data}/"
    nx.write_edgelist(graph, path + f"{args.data}.edges", data=["weight"])
    torch.save(labels, path + f"{args.data}-labels.pt")
