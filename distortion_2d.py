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
import numpy as np
from types import SimpleNamespace
from statistics import mean
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
import math
from tqdm import tqdm
from sympa import config
from sympa.metrics import AverageDistortionMetric
from sympa.model import Model
import sympa.math.symmetric_math as sm
from distortion_histogram import load_model
from train import load_training_data
from random import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def get_avg_distortion(model, src_dst_ids, graph_distances):
    triplets = TensorDataset(src_dst_ids, graph_distances)
    eval_split = DataLoader(triplets, sampler=SequentialSampler(triplets), batch_size=4096)
    distortion_metric = AverageDistortionMetric()
    model.eval()
    total_distortion = []
    for batch in tqdm(eval_split, desc="Evaluating"):
        src_dst_ids, graph_distances = batch
        with torch.no_grad():
            manifold_distances = model(src_dst_ids)
            distortion = distortion_metric.calculate_metric(graph_distances, manifold_distances)
        total_distortion.extend(distortion.tolist())
    return mean(total_distortion)


def build_model_with_embeds(src_model, vector_embeds):
    """
    :param vector_embeds: num_points x 2
    """
    if src_model == "bounded":
        target_model_name = "poincare"
        dims = 2
    else:           # src_model is upper
        target_model_name = "upper"
        dims = 1
        vector_embeds = vector_embeds.view(-1, 2, 1, 1)

    d = {'model': target_model_name, 'dims': dims, 'num_points': len(vector_embeds)}
    args = SimpleNamespace(**d)
    model = Model(args)
    model.embeddings.embeds.data.copy_(vector_embeds)
    return model


def get_points_in_circumference(radius, n_points=100):
    pi = math.pi
    return [(math.cos(2 * pi / n_points * x) * radius,
             math.sin(2 * pi / n_points * x) * radius) for x in range(0, n_points + 1)]


def get_points_in_sphere(radius, n_points=100, z_points=10):
    pi = math.pi
    points = []
    z_ini, z_end = 0, 0.95
    for z in np.arange(z_ini, z_end, (z_end - z_ini) / z_points):
        for i in range(n_points):
            x = math.cos(2 * pi / n_points * i) * radius
            y = math.sin(2 * pi / n_points * i) * radius

            # quizas conviene generar x e y para distintos radios y despejar el z de ahi y ya, sabiendo que la norma es 1
            # o quizas usar esto asi y ya, tampoco es tan terrible.
            # Probar con esto y vemos

            norm = (x**2 + y**2 + z**2)**0.5
            points.append((x / norm, y / norm, z / norm))
    return points


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
    plt.savefig("plots/distortion_2d/" + title + ".png")


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
    parser.add_argument("--ckpt_path", default="ckpt/up3d-grid-best-10ep", required=False, help="Path to model to load")
    parser.add_argument("--data", default="grid2d-36", required=False, type=str, help="Name of prep folder")
    parser.add_argument("--model", default="upper", type=str, help="Name of manifold used in the run")
    parser.add_argument("--scale_triplets", default=0, type=int, help="Whether to apply scaling to triplets or not")
    parser.add_argument("--subsample", default=-1, type=float, help="Subsamples the % of closest triplets")
    parser.add_argument("--scale_init", default=1, type=float, help="Value to init scale.")

    args = parser.parse_args()
    src_model, matrix_embeds = load_model(args)
    id2node, _, _, valid_src_dst_ids, valid_distances = load_training_data(args)
    avg_distortion_src_model = get_avg_distortion(src_model, valid_src_dst_ids, valid_distances)

    xs = []
    ys = []
    zs = []

    n_points = 50
    # points = get_points_inside_circle(min_radius=0.5, max_radius=1, n_points=round(n_points * 0.65))
    points = get_points_in_circumference(radius=1, n_points=round(n_points))
    # points += get_points_in_circumference(radius=0.75, n_points=round(n_points * 0.35))
    # points += get_points_in_circumference(radius=0.5, n_points=round(n_points * 0.2))
    for x, y in points:
        vector_embeds = map_4d_to_2d(matrix_embeds, x, y)
        target_model = build_model_with_embeds(args.model, vector_embeds)

        current_avg_distortion = get_avg_distortion(target_model, valid_src_dst_ids, valid_distances)

        xs.append(x)
        ys.append(y)
        zs.append(current_avg_distortion)

    title = f"{args.model}{args.dims}d-{args.data}_orig_dist{avg_distortion_src_model:.3f}"
    plot3d(xs, ys, zs, title)


if __name__ == "__main__":
    main()
