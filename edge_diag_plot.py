
import argparse
import torch
import random
from tqdm import tqdm
from sympa import config
from sympa.utils import get_logging
from sympa.metrics import AverageDistortionMetric
import torch.distributed as dist
from statistics import mean
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from train import load_training_data, get_model
import numpy as np
import networkx as nx
sns.set()


def get_diag_entries(model, data_loader):
    model.eval()
    src_dst, graph_dists, diag_entries = [], [], []
    total_distortion = []
    metric = AverageDistortionMetric()      # compute distortion as a sanity check
    for batch in tqdm(data_loader, desc="Evaluating"):
        src_dst_ids, graph_distances = batch
        with torch.no_grad():
            manifold_distances, diag = model(src_dst_ids)
            src_dst.append(src_dst_ids)
            graph_dists.append(graph_distances)
            diag_entries.append(diag)
            # sanity check
            distortion = metric.calculate_metric(graph_distances, manifold_distances)
        total_distortion.extend(distortion.tolist())

    src_dst = torch.cat(src_dst, dim=0)
    graph_dists = torch.cat(graph_dists, dim=0)
    diag_entries = torch.cat(diag_entries, dim=0)

    avg_distortion = mean(total_distortion)
    return src_dst, graph_dists, diag_entries, avg_distortion


def plot3d(xs, ys, zs, title):
    cmap = sns.color_palette("viridis_r", as_cmap=True)
    f, ax = plt.subplots()
    points = ax.scatter(xs, ys, c=zs, s=10, cmap=cmap)
    plt.axis('equal')
    f.colorbar(points)
    # plt.title(title)
    # plt.show()
    plt.savefig("plots/edges/" + title + f"-{len(xs)}.png")


def plot2d_axis_paths(src_dst_ids, graph_dists, diag_entries, title):
    root_id = 12
    hor_coords, ver_coords, diag_coords = filter_edges_by_type(diag_entries, src_dst_ids, root_id)

    fig, ax = plt.subplots()
    max_tick = -1
    for coords, name, color in zip([hor_coords, ver_coords, diag_coords], ["hor", "ver", "diag"], ["red", "blue", "green"]):
        xs = [t[0] for t in coords]
        ys = [t[1] for t in coords]
        max_tick = max(*xs, max_tick)
        max_tick = max(*ys, max_tick)
        ax.scatter(xs, ys, c=color, label=name, alpha=0.5)

    step = max_tick / 10
    plt.yticks(np.arange(0, max_tick, step=step))
    plt.xticks(np.arange(0, max_tick, step=step))

    ax.legend()
    # plt.axis('equal')
    ax.grid(True)
    # plt.show()
    plt.savefig("plots/edges/newplots/" + title + f"root{root_id}-axispaths.png")


def filter_edges_by_type(diag_entries, src_dst_ids, root_id=12):
    """
    Given a root, it looks for edges that are located in horizontal, vertical and diagonal axis, from a given root
    """
    hor_ids = {2, 7, 17, 22}
    ver_ids = {10, 11, 13, 14}
    diag_ids = {0, 6, 18, 24, 4, 8, 16, 20}
    hor_coords, ver_coords, diag_coords = [], [], []
    for i in range(len(src_dst_ids)):
        src, dst = src_dst_ids[i].tolist()
        if src != root_id and dst != root_id: continue
        if dst == root_id: dst = src  # tuples are not repeated so sometimes you have to invert it
        if dst in hor_ids:
            hor_coords.append(diag_entries[i].tolist())
        elif dst in ver_ids:
            ver_coords.append(diag_entries[i].tolist())
        elif dst in diag_ids:
            diag_coords.append(diag_entries[i].tolist())
    return hor_coords, ver_coords, diag_coords


def plot_graph_nodes_from_root_by_angle(src_dst_ids, graph_dists, diag_entries, title):
    """
    Graph visualization of the NODES in the graph, from a given root R, to all other vertices.
    Color of the node X represents the angle of the entries of the path (R,X)
    """
    graph = get_graph(graph_dists, src_dst_ids)
    root_id = 0

    graph.nodes[root_id]["angle"] = 0
    node_colors = [(root_id, 0)]
    for i in range(len(src_dst_ids)):
        src, dst = src_dst_ids[i].tolist()
        if src != root_id and dst != root_id: continue
        if dst == root_id: dst = src
        x, y = diag_entries[i]
        r = diag_entries[i].norm()
        angle = torch.acos(x / r).item()   # should be in [0, pi/4 (0.785)]
        node_colors.append((dst, angle))
        graph.nodes[dst]["angle"] = angle
    return graph

    pos = nx.kamada_kawai_layout(graph) # spring_layout(graph, iterations=100)

    # edges = graph.edges()
    # angles = [graph[u][v]['angle'] for u, v in edges]

    # fig = plt.figure()
    cmap = sns.color_palette("viridis_r", as_cmap=True)
    nc = nx.draw_networkx_nodes(graph, pos, node_size=100, nodelist=[t[0] for t in node_colors], node_color=[t[1] for t in node_colors], cmap=cmap)
    nx.draw_networkx_edges(graph, pos, width=1)
    # nx.draw(graph, pos, ax=fig.add_subplot(111), node_size=5, with_labels=True, edges=edges, edge_color=angles, width=5,
    #         cmap=plt.cm.jet)
    # nx.draw(graph, pos, node_size=5, with_labels=True, edgelist=edges, edge_color=angles, width=3,
    #         edge_cmap=plt.cm.jet)
    plt.colorbar(nc)
    plt.axis('off')
    # plt.show()
    plt.savefig("plots/edges/newplots/vizgraph-" + title + f"-root{root_id}.png")


def plot_graph_edges_by_angle(src_dst_ids, graph_dists, diag_entries, title, graph, export_graph=True):
    """
    Graph visualization of the edges in the graph.
    Color of the edge (A,B) represents the angle of the entries of the path (A,B)
    """
    # graph = nx.Graph()
    for i in range(len(src_dst_ids)):
        gdist = graph_dists[i].item()
        if gdist != 1: continue
        src, dst = src_dst_ids[i].tolist()
        x, y = diag_entries[i]
        r = diag_entries[i].norm()
        angle = torch.acos(x / r).item()   # should be in [0, pi/4 (0.785)]
        graph.edges[(src, dst)]["angle"] = angle

    if export_graph:
        nx.write_gexf(graph, path="graphs/" + title + ".gexf")
        return

    pos = nx.kamada_kawai_layout(graph)     # spring_layout(graph, iterations=100)

    edges = graph.edges()
    angles = [graph[u][v]['angle'] for u, v in edges]

    # fig = plt.figure()
    cmap = sns.color_palette("viridis_r", as_cmap=True)
    ec = nx.draw_networkx_edges(graph, pos, width=3, edgelist=edges, edge_color=angles, edge_cmap=cmap)
    nx.draw_networkx_nodes(graph, pos, node_size=30)

    # nx.draw(graph, pos, ax=fig.add_subplot(111), node_size=5, with_labels=True, edges=edges, edge_color=angles, width=5,
    #         cmap=plt.cm.jet)
    # nx.draw(graph, pos, node_size=5, with_labels=True, edgelist=edges, edge_color=angles, width=3,
    #         edge_cmap=plt.cm.jet)
    plt.colorbar(ec)
    plt.axis('off')
    # plt.show()
    plt.savefig("plots/edges/newplots/vizgraph-" + title + f"-edges.png")


def get_graph(graph_dists, src_dst_ids):
    graph = nx.Graph()
    for i in range(len(graph_dists)):
        gdist = graph_dists[i].item()
        if gdist == 1:
            src, dst = src_dst_ids[i].tolist()
            graph.add_edge(src, dst)
    return graph


def plot2d_all_edges(src_dst_ids, graph_dists, diag_entries, title):
    """
    Plots in 2d the coords of all edges from a given root.
    """
    diag_coords = []
    for i in range(len(graph_dists)):
        gdist = graph_dists[i].item()
        if gdist != 1: continue
        diag_coords.append(diag_entries[i].tolist())

    fig, ax = plt.subplots()
    max_tick = -1
    xs = [t[0] for t in diag_coords]
    ys = [t[1] for t in diag_coords]
    max_tick = max(*xs, max_tick)
    max_tick = max(*ys, max_tick)
    points = ax.scatter(xs, ys)

    step = max_tick / 10
    plt.yticks(np.arange(0, max_tick, step=step))
    plt.xticks(np.arange(0, max_tick, step=step))

    # for i in range(len(diag_coords)):
    #     ax.annotate(str(dsts[i]), (xs[i], ys[i]))

    # ax.legend()
    # plt.axis('equal')
    # fig.colorbar(points)
    ax.grid(True)
    # plt.show()
    plt.savefig("plots/edges/newplots/" + title + f"-allEdges.png")


def main():
    parser = argparse.ArgumentParser(description="edge_diag_plot.py")
    parser.add_argument("--load_model", default="ckpt/upper-fone2d-grid2d-81-best-975ep", required=False, help="Path to model to load")
    # parser.add_argument("--data", default="prod-cart-treetree", required=False, type=str, help="Name of prep folder")
    # parser.add_argument("--model", default="upper-fone", type=str, help="Name of manifold used in the run")
    parser.add_argument("--dims", default=2, type=int, help="Dimensions for the model.")
    parser.add_argument("--scale_init", default=1, type=float, help="Value to init scale.")
    parser.add_argument("--scale_coef", default=1, type=float, help="Coefficient to divide scale.")
    parser.add_argument("--train_scale", default=0, type=int, help="Whether to train scaling or not.")
    # optim and config
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--scale_triplets", default=0, type=int, help="Whether to apply scaling to triplets or not")
    parser.add_argument("--subsample", default=1, type=float, help="Subsamples the % of closest triplets")
    parser.add_argument("--plot_subsample", default=1, type=float, help="Subsamples the % of closest triplets")

    parser.add_argument("--local_rank", type=int, help="Local process rank assigned by torch.distributed.launch")
    parser.add_argument("--n_procs", default=1, type=int, help="Number of process to create")

    args = parser.parse_args()

    # get data:
    if "prod-cart-treetree" in args.load_model: args.data = "prod-cart-treetree"
    if "prod-cart-treegrid2d-large" in args.load_model: args.data = "prod-cart-treegrid2d-large"
    if "prod-root-gridtrees9-2-5" in args.load_model: args.data = "prod-root-gridtrees9-2-5"
    if "prod-root-treegrids16-2-4" in args.load_model:args.data = "prod-root-treegrids16-2-4"
    if "prod-root-treegrids-9-2-2" in args.load_model: args.data = "prod-root-treegrids-9-2-2"
    if "prod-root-gridtrees-9-2-2" in args.load_model: args.data = "prod-root-gridtrees-9-2-2"
    if "grid2d-25" in args.load_model: args.data = "grid2d-25"
    if "grid4d-256" in args.load_model: args.data = "grid4d-256"
    if "grid4d-625" in args.load_model: args.data = "grid4d-625"
    if "grid2d-81" in args.load_model: args.data = "grid2d-81"
    if "tree3-5" in args.load_model: args.data = "tree3-5"
    if "tree3-3" in args.load_model: args.data = "tree3-3"
    if "bio-diseasome" in args.load_model: args.data = "bio-diseasome"
    if "csphd" in args.load_model: args.data = "csphd"
    if "road-euroroad" in args.load_model: args.data = "road-euroroad"
    if "usca312" in args.load_model: args.data = "usca312"
    if "facebook" in args.load_model: args.data = "facebook"

    # get model
    if "upper" in args.load_model: args.model = "upper"
    if "bounded" in args.load_model: args.model = "upper"
    if "upper" in args.load_model and "finsler" in args.load_model: args.model = "upper-fone"
    if "bounded" in args.load_model and "finsler" in args.load_model: args.model = "bounded-fone"
    if "upper" in args.load_model and "fininf" in args.load_model: args.model = "upper-finf"
    if "bounded" in args.load_model and "fininf" in args.load_model: args.model = "bounded-finf"
    if "upper" in args.load_model and "fone" in args.load_model: args.model = "upper-fone"
    if "bounded" in args.load_model and "fone" in args.load_model: args.model = "bounded-fone"
    if "upper" in args.load_model and "finf" in args.load_model: args.model = "upper-finf"
    if "bounded" in args.load_model and "finf" in args.load_model: args.model = "bounded-fone"

    log = get_logging()
    if args.local_rank == 0:
        log.info(args)

    dist.init_process_group(backend=config.BACKEND, init_method='env://') # world_size=args.n_procs, rank=args.local_rank)

    id2node, train_loader, _ = load_training_data(args, log)
    args.num_points = len(id2node)
    model = get_model(args)

    src_dst_ids, graph_dists, diag_entries, avg_distortion = get_diag_entries(model, train_loader)
    index = torch.LongTensor(random.sample(list(range(len(src_dst_ids))), round(len(src_dst_ids) * args.plot_subsample)))
    log.info(f"Average distortion over training set: {avg_distortion * 100:.2f}")

    title = f"{args.model}{args.dims}d-{args.data}"
    graph = plot_graph_nodes_from_root_by_angle(src_dst_ids, graph_dists, diag_entries, title)
    plt.clf()
    plot_graph_edges_by_angle(src_dst_ids, graph_dists, diag_entries, title, graph)
    plt.clf()
    plot2d_all_edges(src_dst_ids, graph_dists, diag_entries, title)
    # plt.clf()
    return

    xs = diag_entries[index, 0].numpy()
    ys = diag_entries[index, 1].numpy()
    zs = graph_dists[index].numpy()
    log.info(title)
    plot3d(xs, ys, zs, title)
    # plt.savefig("plots/distor/" + title + ".png")


if __name__ == "__main__":
    main()
