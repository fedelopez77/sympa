"""
Here the idea is the following:
1 - Take 2 x 2 complex symmetric matrices
2 - Choose v1, v2 such that the vector (v1, v2) has norm = 1
3 - Multiply: (v1 v2) [(a b), (b c)] (v1 v2)^T.
    This gives a complex number z
4 - Take x = real(z), y = imag(z), and assume that (x, y) is a point in the Poincare ball of dim = 2 (H^2)
5- Measure the distortion of the resulting projected points in H^2
6 - Plot the distortion values for different values of v1, v2
"""

import argparse
import torch
from types import SimpleNamespace
from statistics import mean
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import math
from tqdm import tqdm
from sympa import config
from sympa.metrics import AverageDistortionMetric
from sympa.model import Model
import sympa.math.symmetric_math as sm
from random import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def load_model(args):
    data = torch.load(args.ckpt_path)
    model_state, id2node = data["model"], data["id2node"]
    embeds = model_state["embeddings.embeds"]
    args.num_points = len(embeds)
    args.dims = embeds.shape[-1]
    model = Model(args)
    model.load_state_dict(model_state)
    return model, embeds


def load_prep(data_path):
    data_path = config.PREP_PATH / f"{data_path}/{config.PREPROCESSED_FILE}"
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    id2node = data["id2node"]
    triplets = torch.LongTensor(list(data["triplets"])).to(config.DEVICE)
    return triplets, id2node


def get_avg_distortion(model, triplets):
    triplets = TensorDataset(triplets)
    eval_split = DataLoader(triplets, sampler=SequentialSampler(triplets), batch_size=1024)
    distortion_metric = AverageDistortionMetric()
    model.eval()
    total_distortion = []
    for batch in tqdm(eval_split, desc="Evaluating"):
        batch_points = batch[0].to(config.DEVICE)
        with torch.no_grad():
            manifold_distances = model(batch_points)
            graph_distances = batch_points[:, -1]
            distortion = distortion_metric.calculate_metric(graph_distances, manifold_distances)
        total_distortion.extend(distortion.tolist())
    return mean(total_distortion)


def get_poincare_model(vector_embeds):
    d = {'model': 'poincare', 'dims': 2, 'num_points': len(vector_embeds)}
    args = SimpleNamespace(**d)
    poincare_model = Model(args)
    poincare_model.embeddings.embeds.data.copy_(vector_embeds)
    return poincare_model


def get_points_in_circumference(radius, n_points=100):
    pi = math.pi
    return [(math.cos(2 * pi / n_points * x) * radius,
             math.sin(2 * pi / n_points * x) * radius) for x in range(0, n_points + 1)]


def get_points_inside_circle(min_radius, max_radius, n_points=100):
    def get_random_point_in_circle(min_radius, max_radius):
        t = 2 * math.pi * random()
        radius = min_radius + (max_radius - min_radius) * random()
        return radius * math.cos(t), radius * math.sin(t)
    return [get_random_point_in_circle(min_radius, max_radius) for _ in range(n_points)]


def plot3d(xs, ys, zs, title):
    cmap = sns.color_palette("viridis_r", as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(xs, ys, c=zs, s=50, cmap=cmap)
    f.colorbar(points)
    plt.title(title)
    # plt.show()
    plt.savefig("plots/poincareDist/" + title + ".png")


def map_4d_to_2d(matrix_embeds, x, y):
    """
    Applies multiplication of (x y) A (x y)^T.
    Maps the real and imaginary part of the result to a 2d vector

    :param matrix_embeds: num_points x 2 x 2 x 2
    :param x: float: real value
    :param y: float: real value
    :return: num_points x 2 tensor
    """
    v_hor = torch.zeros((1, 2, 2, 2))
    v_hor[0, 0, 0, 0] = x
    v_hor[0, 0, 0, 1] = y
    v_ver = sm.transpose(v_hor)

    v_hor = v_hor.repeat(len(matrix_embeds), 1, 1, 1)
    v_ver = v_ver.repeat(len(matrix_embeds), 1, 1, 1)
    res = sm.bmm3(v_hor, matrix_embeds, v_ver)  # n x 2 x 2 x 2

    real_coord = torch.unsqueeze(res[:, 0, 0, 0], 1)
    imag_coord = torch.unsqueeze(res[:, 1, 0, 0], 1)
    vecs_2d = torch.cat((real_coord, imag_coord), dim=-1)
    return vecs_2d


def main():
    parser = argparse.ArgumentParser(description="plot_2d_distortion.py")
    parser.add_argument("--ckpt_path", default="ckpt/upper2d-grid2d-best-500ep", required=False, help="Path to model to load")
    parser.add_argument("--data", default="grid2d-36", required=False, type=str, help="Name of prep folder")
    parser.add_argument("--model", default="upper", type=str, help="Name of manifold used in the run")

    args = parser.parse_args()
    model, matrix_embeds = load_model(args)
    triplets, id2node = load_prep(args.data)
    # avg_distortion_original_model = get_avg_distortion(model, triplets)

    xs = []
    ys = []
    zs = []

    n_points = 100
    # points = get_points_inside_circle(min_radius=0.5, max_radius=1, n_points=round(n_points * 0.65))
    points = get_points_in_circumference(radius=1, n_points=round(n_points * 0.45))
    points += get_points_in_circumference(radius=0.75, n_points=round(n_points * 0.35))
    points += get_points_in_circumference(radius=0.5, n_points=round(n_points * 0.2))
    for x, y in points:
        vector_embeds = map_4d_to_2d(matrix_embeds, x, y)
        poincare_model = get_poincare_model(vector_embeds)

        current_avg_distortion = get_avg_distortion(poincare_model, triplets)

        xs.append(x)
        ys.append(y)
        zs.append(current_avg_distortion)

    title = f"{args.model}{args.dims}d-{args.data}"
    plot3d(xs, ys, zs, title)


if __name__ == "__main__":
    main()
