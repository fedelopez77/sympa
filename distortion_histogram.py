
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
sns.set()


def load_model(args):
    data = torch.load(args.ckpt_path)
    model_state, id2node = data["model"], data["id2node"]
    embeds = model_state["embeddings.embeds"]
    args.num_points = len(embeds)
    args.dims = embeds.shape[-1]
    model = Model(args)
    model.load_state_dict(model_state)
    return model


def load_prep(data_path):
    data_path = config.PREP_PATH / f"{data_path}/{config.PREPROCESSED_FILE}"
    print(f"Loading data from {data_path}")
    data = torch.load(data_path)
    id2node = data["id2node"]
    triplets = torch.LongTensor(list(data["triplets"])).to(config.DEVICE)
    return triplets, id2node


def get_distortion(model, triplets):
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
    return total_distortion


def main():
    parser = argparse.ArgumentParser(description="distortion_histogram.py")
    parser.add_argument("--ckpt_path", default="ckpt/euclid-grid2d-best-500ep", required=False, help="Path to model to load")
    parser.add_argument("--data", default="grid2d-36", required=False, type=str, help="Name of prep folder")
    parser.add_argument("--model", default="euclidean", type=str, help="Name of manifold used in the run")

    args = parser.parse_args()
    model = load_model(args)
    triplets, id2node = load_prep(args.data)

    distortion = get_distortion(model, triplets)

    plt.hist(distortion, bins=30)
    title = f"{args.model}{args.dims}d-{args.data}"
    plt.title(title)
    plt.xlabel("Distortion")
    plt.savefig("plots/distor/" + title + ".png")


if __name__ == "__main__":
    main()
