
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
    plt.title(title)
    # plt.show()
    plt.savefig("plots/edges/" + title + ".png")


def main():
    parser = argparse.ArgumentParser(description="distortion_histogram.py")
    parser.add_argument("--load_model", default="ckpt/rupper2-prod-cart-treetree-finsler-lr1e-2-mgr10-bs128-2-best-1000ep", required=False, help="Path to model to load")
    # parser.add_argument("--data", default="prod-cart-treetree", required=False, type=str, help="Name of prep folder")
    # parser.add_argument("--model", default="upper-fone", type=str, help="Name of manifold used in the run")
    parser.add_argument("--dims", default=2, type=int, help="Dimensions for the model.")
    parser.add_argument("--scale_init", default=1, type=float, help="Value to init scale.")
    parser.add_argument("--scale_coef", default=1, type=float, help="Coefficient to divide scale.")
    parser.add_argument("--train_scale", default=0, type=int, help="Whether to train scaling or not.")
    # optim and config
    parser.add_argument("--batch_size", default=1000, type=int, help="Batch size.")
    parser.add_argument("--scale_triplets", default=0, type=int, help="Whether to apply scaling to triplets or not")
    parser.add_argument("--subsample", default=0.5, type=float, help="Subsamples the % of closest triplets")
    parser.add_argument("--plot_subsample", default=0.1, type=float, help="Subsamples the % of closest triplets")

    parser.add_argument("--local_rank", type=int, help="Local process rank assigned by torch.distributed.launch")
    parser.add_argument("--n_procs", default=1, type=int, help="Number of process to create")

    args = parser.parse_args()

    # get data:
    if "prod-cart-treetree" in args.load_model: args.data = "prod-cart-treetree"
    if "prod-root-gridtrees9-2-5" in args.load_model: args.data = "prod-root-gridtrees9-2-5"
    if "prod-root-treegrids16-2-4" in args.load_model:args.data = "prod-root-treegrids16-2-4"
    if "grid4d-256" in args.load_model: args.data = "grid4d-256"
    if "tree3-5" in args.load_model: args.data = "tree3-5"
    if "bio-diseasome" in args.load_model: args.data = "bio-diseasome"
    if "csphd" in args.load_model: args.data = "csphd"
    if "road-euroroad" in args.load_model: args.data = "road-euroroad"
    if "usca312" in args.load_model: args.data = "usca312"

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

    # hacer pasada y collect lista de python con src,dst,dist,diag_entries
    src_dst_ids, graph_dists, diag_entries, avg_distortion = get_diag_entries(model, train_loader)
    index = torch.LongTensor(random.sample(list(range(len(src_dst_ids))), round(len(src_dst_ids) * args.plot_subsample)))

    log.info(f"Average distortion over training set: {avg_distortion * 100:.2f}")
    xs = diag_entries[index, 0].numpy()
    ys = diag_entries[index, 1].numpy()
    zs = graph_dists[index].numpy()
    title = f"{args.model}{args.dims}d-{args.data}-sample{len(xs)}"
    log.info(title)
    plot3d(xs, ys, zs, title)
    # plt.savefig("plots/distor/" + title + ".png")


if __name__ == "__main__":
    main()
