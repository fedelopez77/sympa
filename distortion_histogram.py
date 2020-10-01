
import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from sympa import config
from sympa.metrics import AverageDistortionMetric
from sympa.model import Model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from train import load_training_data
sns.set()


def load_model(args):
    data = torch.load(args.ckpt_path)
    model_state, id2node = data["model"], data["id2node"]
    embeds = model_state["embeddings.embeds"]
    args.num_points = len(embeds)
    args.dims = embeds.shape[-1]
    args.scale_coef = args.scale_init = 1
    args.train_scale = False
    model = Model(args)
    model.load_state_dict(model_state)
    model.scale.data = torch.DoubleTensor([1])
    model.scale_coef = 1
    return model, embeds


def get_distortion(model, src_dst_ids, graph_distances):
    triplets = TensorDataset(src_dst_ids, graph_distances)
    eval_split = DataLoader(triplets, sampler=SequentialSampler(triplets), batch_size=1024)
    distortion_metric = AverageDistortionMetric()
    model.eval()
    total_distortion = []
    for batch in tqdm(eval_split, desc="Evaluating"):
        src_dst_ids, graph_distances = batch
        with torch.no_grad():
            manifold_distances = model(src_dst_ids)
            # distortion = distortion_metric.calculate_metric(graph_distances, manifold_distances)
            distortion = manifold_distances / graph_distances   # Modified formula upon request
        total_distortion.extend(distortion.tolist())
    return total_distortion


def main():
    parser = argparse.ArgumentParser(description="distortion_histogram.py")
    parser.add_argument("--ckpt_path", default="ckpt/up2d-tree-coef10-best-995ep", required=False, help="Path to model to load")
    parser.add_argument("--data", default="tree3-3", required=False, type=str, help="Name of prep folder")
    parser.add_argument("--model", default="upper", type=str, help="Name of manifold used in the run")
    parser.add_argument("--scale_triplets", default=0, type=int, help="Whether to apply scaling to triplets or not")
    parser.add_argument("--subsample", default=-1, type=float, help="Subsamples the % of closest triplets")
    parser.add_argument("--scale_init", default=1, type=float, help="Value to init scale.")

    args = parser.parse_args()
    model, _ = load_model(args)
    id2node, _, _, valid_src_dst_ids, valid_distances = load_training_data(args)

    distortion = get_distortion(model, valid_src_dst_ids, valid_distances)

    plt.hist(distortion, bins=30)
    title = f"{args.model}{args.dims}d-{args.data}"
    plt.title(title)
    plt.xlabel("Distortion")
    # plt.show()
    plt.savefig("plots/distor/" + title + ".png")


if __name__ == "__main__":
    main()
